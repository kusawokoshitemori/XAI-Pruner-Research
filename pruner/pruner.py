import torch
import numpy as np
import random
import timm
from torch.ao.nn.quantized.functional import threshold

from model import VisionTransformer
from lrp.module import ViTLRP, Conv2dLRP, BasicBlockLRP, BottleneckLRP, EpsilonLinearLRP

def get_scores(model):
    if isinstance(model, ViTLRP):
        return get_scores_vit(model)
    elif isinstance(model, timm.models.ResNet):
        return get_scores_resnet(model)
    elif isinstance(model, timm.models.VGG):
        return get_scores_vgg(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

def update_scores(model, scores, momentum: float):
    if isinstance(model, ViTLRP):
        return update_scores_vit(model, scores, momentum)

    elif isinstance(model, timm.models.ResNet):
        return update_scores_resnet(model, scores, momentum)

    elif isinstance(model, timm.models.VGG):
        return update_scores_vgg(model, scores, momentum)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

def create_crntroller(model, scores, pruning_rate, population=50, epsilon=0.02, protect=0.0,epochs=50):
    if isinstance(model, VisionTransformer):
        return Controller_vit(model, scores, pruning_rate, population, epsilon, protect, epochs)
    elif isinstance(model, timm.models.ResNet):
        return Controller_resnet(model, scores, pruning_rate, population, epsilon, protect, epochs)
    elif isinstance(model, timm.models.VGG):
        return Controller_vgg(model, scores, pruning_rate, population, epsilon, protect, epochs)

def get_scores_vit(model:ViTLRP):
    scores = dict()
    relevance = model.get_relevance()

    scores["head"], scores["hidden"], dim_score = list(), list(), list()

    for i in range(model.depth):
        scores["head"].append(torch.mean(torch.sum(torch.abs(relevance["head"][i]), dim=(2,3)), dim=0).cpu().numpy())
        scores["hidden"].append(torch.mean(torch.sum(torch.abs(relevance["hidden"][i]), dim=1), dim=0).cpu().numpy())

    scores["dim"] = torch.mean(torch.sum(torch.abs(relevance["dim"]), dim=1), dim=0).cpu().numpy()
    scores["head"] = np.asarray(scores["head"])
    scores["hidden"] = np.asarray(scores["hidden"])

    model.clear_relevance()
    return scores

def get_scores_resnet(model:timm.models.ResNet):

    conv1_relevance = model.conv1.get_relevance()

    def get_relevance(model, relevance):
        for name, child in model.named_children():
            if isinstance(child, BasicBlockLRP):
               relevance.append(child.get_relevance())
            elif isinstance(child, BottleneckLRP):
                relevance.append(child.get_relevance())
            else:
                if any(child.named_children()):
                    get_relevance(child, relevance)


    def clear_relevance(model):
        for name, child in model.named_children():
            if isinstance(child, Conv2dLRP):
                child.clear_relevance()
            else:
                if any(child.named_children()):
                    clear_relevance(child)

    relevance = list()
    get_relevance(model, relevance)
    scores = dict()
    i = 0
    scores['conv1'] = [conv1_relevance]
    for block_relevance in relevance:
        name = 'block' + str(i)
        scores[name] = block_relevance
        i += 1

    clear_relevance(model)

    return scores

def get_scores_vgg(model:timm.models.VGG):
    scores = dict()
    scores['conv'] = list()

    def get_layer_scores(model, scores):
        for name, child in model.named_children():
            if isinstance(child, Conv2dLRP):
               scores['conv'].append(child.get_relevance())
            else:
                if any(child.named_children()):
                    get_layer_scores(child, scores)

    get_layer_scores(model, scores)
    return scores

def update_scores_vit(model:ViTLRP, scores, momentum:float):
    if scores is None:
        scores = get_scores(model)
    else:
        new_scores = get_scores(model)
        scores["dim"] = scores["dim"] * momentum + new_scores["dim"]
        scores["head"] = scores["head"] * momentum + new_scores["head"]
        scores["hidden"] = scores["hidden"] * momentum + new_scores["hidden"]

    return scores

def update_scores_resnet(model:timm.models.ResNet, scores, momentum:float):
    if scores is None:
        scores = get_scores(model)
    else:
        new_scores = get_scores(model)

        scores['conv1'] = np.asarray(scores['conv1']) * momentum + np.asarray(new_scores['conv1'])

        for i in range(len(scores) -1 ):
            name = "block" + str(i)
            for j in range(len(scores[name])):
                scores[name][j] = np.asarray(scores[name][j]) * momentum + np.asarray(new_scores[name][j])

    return scores

def update_scores_vgg(model:timm.models.VGG, scores, momentum:float):
    if scores is None:
        scores = get_scores(model)
    else:
        new_scores = get_scores(model)
        for i in range(len(scores["conv"])):
            scores['conv'][i] = scores['conv'][i] * momentum + new_scores["conv"][i]

    return scores


class Controller_vit(object):
    def __init__(self,
                 model: ViTLRP,
                 scores,
                 pruning_rate,
                 population=50,
                 epsilon=0.02,
                 protect=0.0,
                 epochs=50):
        self.model = model
        self.scores = scores

        self.embed_dim = model.embed_dim
        self.hidden_dim = model.hidden_dim * model.depth
        self.num_heads = model.num_heads * model.depth
        self.hidden_dim_per_layer = model.hidden_dim
        self.num_heads_per_layer = model.num_heads
        self.depth = model.depth

        self.candidates = set()
        self.population = population
        self.epochs = epochs

        self.percentage = pruning_rate
        self.epsilon = epsilon
        self.protect_percent = protect

        self.original_flops = self.model.get_complexity(dim_rate=0.0, hidden_rate=0.0, head_rate=0.0)

        self.masks = None
        self.best = None


    def calculate_fitness(self, candidates):
        fitness = list()

        head_flatten_scores = np.sort(np.abs(np.asarray([score for layer in self.scores["head"] for score in layer])))
        hidden_flatten_scores = np.sort(np.abs(np.asarray([score for layer in self.scores["hidden"] for score in layer])))
        dim_flatten_scores = np.sort(np.abs(np.asarray(self.scores["dim"])))

        for candidate in candidates:
            dim_rate, hidden_rate, head_rate = candidate[0], candidate[1], candidate[2]

            pruned_dim = int(self.embed_dim * dim_rate)
            pruned_hidden = int(self.hidden_dim * hidden_rate)
            pruned_head = int(self.num_heads * head_rate)

            total_scores = (np.sum(head_flatten_scores[:pruned_head]) +
                            np.sum(hidden_flatten_scores[:pruned_hidden]) +
                            np.sum(dim_flatten_scores[:pruned_dim]))
            fitness.append(total_scores)
        return fitness


    def is_legal(self, candidate):
        if candidate[0] > 0 and candidate[1] > 0 and candidate[2] > 0:
            current_flops = self.model.get_complexity(dim_rate=candidate[0], hidden_rate=candidate[1], head_rate=candidate[2])
            pruned_percentage = 1 - ( current_flops / self.original_flops)
            if  self.percentage - self.epsilon < pruned_percentage < self.percentage + self.epsilon:
                return True
            else:
                return False
        else:
            return False


    def get_random_candidates(self):
        dim_rate, head_rate = np.random.uniform(low=self.percentage, high=1, size=2)
        hidden_rate = 0

        current_flops = self.model.get_complexity(dim_rate, hidden_rate, head_rate)
        pruned_percentage = 1 - (current_flops / self.original_flops)

        while pruned_percentage > self.percentage + self.epsilon:
            head_rate = head_rate * 0.8
            dim_rate = dim_rate * 0.8

            current_flops = self.model.get_complexity(dim_rate=dim_rate, hidden_rate=hidden_rate, head_rate=head_rate)
            pruned_percentage = 1 - (current_flops / self.original_flops)

            if pruned_percentage < self.percentage + self.epsilon:
                start = 0
                end = 1
                mid = 0.5

                while start <= end:
                    current_flops = self.model.get_complexity(dim_rate=dim_rate, hidden_rate=mid, head_rate=head_rate)
                    pruned_percentage = 1 - (current_flops / self.original_flops)

                    if pruned_percentage < self.percentage - self.epsilon and start <= mid:
                        start = mid
                        mid = (start + end) / 2
                    elif pruned_percentage > self.percentage + self.epsilon:
                        end = mid
                        mid = (start + end) / 2
                    else:
                        hidden_rate = mid
                        break
                break
        cand = [dim_rate, hidden_rate, head_rate]
        return cand


    def crossover(self, crossover_num, max_iters=100):
        kids = set()
        iters = 0

        while len(kids) < crossover_num and iters < max_iters:
            parent1, parent2 = random.sample(self.candidates, 2)

            mask = np.random.randint(0, 2, size=len(parent1))
            child = np.where(mask == 0, parent1, parent2)

            if self.is_legal(child):
                kids.add(tuple(child))
            iters += 1

        return kids


    def mutation(self, mutation_num, max_iters=100, prob=0.5):
        kids = set()
        iters = 0
        while len(kids) < mutation_num and iters < max_iters:
            cand = np.array(random.sample(self.candidates, 1)[0])

            for i in range(len(cand)):
                random_s = np.random.random()
                if random_s < prob:
                    cand[i] = cand[i] + np.random.uniform(-0.1, 0.1)

            if self.is_legal(cand):
                kids.add(tuple(cand))
            iters += 1

        return kids


    def select(self, top_k=50):
        candidates = list(self.candidates)
        fitness = np.asarray(self.calculate_fitness(candidates))
        self.candidates.clear()

        sorted_idx = np.argsort(fitness)[:top_k]
        selected_candidates = [candidates[index] for index in sorted_idx ]
        self.best = selected_candidates[0]

        self.candidates = set(selected_candidates)

        pruned_dim = int(self.embed_dim * self.best[0])
        pruned_hidden = int(self.hidden_dim * self.best[1])
        pruned_heads = int(self.num_heads * self.best[2])

        print("embed_dim{}  hidden_dim{}  heads{}".format(pruned_dim, pruned_hidden, pruned_heads))

        return

    def generate_masks(self):
        dim_rate, hidden_rate, head_rate = self.best[0], self.best[1], self.best[2]
        pruned_dim = int(self.embed_dim * dim_rate)
        pruned_hidden = int(self.hidden_dim * hidden_rate)
        pruned_head = int(self.num_heads * head_rate)

        head_flatten_scores = np.asarray([score for layer in self.scores["head"] for score in layer])
        hidden_flatten_scores = np.asarray([score for layer in self.scores["hidden"] for score in layer])
        dim_flatten_scores = np.asarray(self.scores["dim"])

        pruned_head_indices = np.argsort(head_flatten_scores)[:pruned_head]
        head_mask = np.ones_like(head_flatten_scores)
        head_mask[pruned_head_indices] = 0
        head_mask = head_mask.reshape((self.depth, self.num_heads_per_layer))

        pruned_hidden_indices = np.argsort(hidden_flatten_scores)[:pruned_hidden]
        hidden_mask = np.ones_like(hidden_flatten_scores)
        hidden_mask[pruned_hidden_indices] = 0
        hidden_mask = hidden_mask.reshape((self.depth, self.hidden_dim_per_layer))

        pruned_dim_indices = np.argsort(dim_flatten_scores)[:pruned_dim]
        dim_mask = np.ones_like(dim_flatten_scores)
        dim_mask[pruned_dim_indices] = 0

        self.masks = dict()
        self.masks["heads"] = head_mask
        self.masks["hidden_dim"] = hidden_mask
        self.masks["embed_dim"] = dim_mask
        return

    def get_mask(self):
        return self.masks

    def engine(self, max_iters=100, crossover_num=25, mutation_num=25, prob=0.25, top_k=50):
        self.scores = self.protect()
        epoch = 0
        while epoch < self.epochs:
            # Init self.candidates
            while len(self.candidates) < self.population:
                cand = self.get_random_candidates()
                self.candidates.add(tuple(cand))

            corssover_kids = self.crossover(crossover_num, max_iters)
            mutation_kids = self.mutation(mutation_num, max_iters, prob)

            self.candidates = self.candidates | corssover_kids | mutation_kids

            self.select(top_k=top_k)

            epoch += 1

        self.generate_masks()

    def protect(self):
        protect_head_per_layer = np.ceil(self.protect_percent * self.num_heads_per_layer).astype(int)
        #protect_hidden_per_layer = np.ceil(self.protect_percent * self.hidden_dim_per_layer).astype(int)

        sorted_head = np.argsort(self.scores["head"], axis=1)
        #sorted_hidden_per_layer = np.argsort(self.scores["hidden"], axis=1)

        for i in range(self.depth):
            protect_head_indices = sorted_head[i][-protect_head_per_layer:]
            self.scores["head"][i][protect_head_indices] = np.inf

            #protect_hidden_indices = sorted_hidden_per_layer[i][-protect_hidden_per_layer:]
            #self.scores["hidden"][i][protect_hidden_indices] = np.inf

        return self.scores

class Controller_resnet(object):
    def __init__(self,
                 model: timm.models.ResNet,
                 scores,
                 pruning_rate,
                 population=50,
                 epsilon=0.02,
                 protect=0.0,
                 epochs=50):

        self.model = model
        self.scores = scores

        self.block_dim = np.asarray([64, 64, 128, 256, 512])

        depth = len(self.scores) - 1
        if depth == 8:
            self.block_num = np.asarray([1, 2, 2, 2 ,2])  # resnet-18
        elif depth == 16:
            self.block_num = np.asarray([1, 3, 4, 6, 3])  # resnet-50

        self.candidates = set()
        self.population = population
        self.epochs = epochs

        self.percentage = pruning_rate
        self.epsilon = epsilon
        self.protect_percent = protect

        self.original_channels = None
        self.dim_per_block = None

        self.masks = None
        self.best = None
        self.init()

    def init(self):
        self.dim_per_block = list()
        for key, value in self.scores.items():
            dim = 0
            for arr in value:
                dim += arr.shape[0]
            self.dim_per_block.append(dim)

        self.dim_per_block = np.asarray(self.dim_per_block)
        self.original_channels = np.sum(self.dim_per_block)

    def calculate_fitness(self, candidates):
        fitness = list()

        flatten_scores = list()
        for key, value in self.scores.items():
            flatten_scores.append(np.sort(np.concatenate(value)))

        for candidate in candidates:
            pruning_rate = np.asarray(candidate)
            pruning_channels = np.round(self.dim_per_block * pruning_rate).astype(int)
            total_scores = 0.0

            for i in range(len(flatten_scores)):
                total_scores +=  np.sum(flatten_scores[i][:pruning_channels[i]])
            fitness.append(total_scores)

        return fitness

    def calculate_pruned_channels(self, candidate):
        pruning_rate = np.asarray(candidate)
        return np.sum(np.round(self.dim_per_block * pruning_rate).astype(int))

    def calculate_pruned_percentage(self, candidate):
        pruned_channel = self.calculate_pruned_channels(candidate)
        return pruned_channel / self.original_channels

    def is_legal(self, candidate):
        pruning_rate = np.asarray(candidate)
        if  np.all(pruning_rate) > 0:
            pruned_percentage = self.calculate_pruned_percentage(candidate)
            if  self.percentage - self.epsilon < pruned_percentage < self.percentage + self.epsilon:
                return True
            else:
                return False
        else:
            return False

    # revise the cand to make it legal
    def revise(self, candidate):
        if np.all(candidate) > 0:
            pruned_channels = self.calculate_pruned_channels(candidate)
            target_channels = int(self.original_channels * self.percentage)
            diff = target_channels - pruned_channels

            rate = diff / (self.dim_per_block[-1] * 2)
            candidate[-2:] = candidate[-2:] + rate
            return candidate

        else:
            return None

    def crossover(self, crossover_num, max_iters=100):
        kids = set()
        iters = 0

        while len(kids) < crossover_num and iters < max_iters:
            parent1, parent2 = random.sample(self.candidates, 2)

            mask = np.random.randint(0, 2, size=len(parent1))
            child = np.where(mask == 0, parent1, parent2)

            if self.is_legal(child):
                kids.add(tuple(child))
            else:
                child = self.revise(child)
                if child is not None:
                    kids.add(tuple(child))
            iters += 1

        return kids

    def mutation(self, mutation_num, max_iters=100, prob=0.5):
        kids = set()
        iters = 0
        while len(kids) < mutation_num and iters < max_iters:
            cand = np.array(random.sample(self.candidates, 1)[0])

            for i in range(len(cand)):
                random_s = np.random.random()
                if random_s < prob:
                    cand[i] = cand[i] + np.random.uniform(-0.1, 0.1)

            if self.is_legal(cand):
                kids.add(tuple(cand))
            else:
                cand = self.revise(cand)
                if cand is not None:
                    kids.add(tuple(cand))
            iters += 1

        return kids

    def select(self, top_k=50):
        candidates = list(self.candidates)
        fitness = np.asarray(self.calculate_fitness(candidates))
        self.candidates.clear()

        sorted_idx = np.argsort(fitness)[:top_k]
        selected_candidates = [candidates[index] for index in sorted_idx ]
        self.best = selected_candidates[0]

        self.candidates = set(selected_candidates)
        return

    def get_random_candidates(self):

        def generate_random_values_with_limits(n, upper_bounds, randomness=0.25):
            parts = len(upper_bounds)

            random_weights = np.random.rand(parts)
            initial_weights = (1 - randomness) * (
                        np.array(upper_bounds) / np.sum(upper_bounds)) + randomness * random_weights
            initial_weights /= np.sum(initial_weights)

            initial_values = (initial_weights * n).astype(int)
            values = np.clip(initial_values, 0, upper_bounds)


            remaining = n - np.sum(values)

            while remaining > 0:
                capacities = upper_bounds - values
                non_full_indices = np.where(capacities > 0)[0]

                if len(non_full_indices) == 0:
                    break

                chosen_idx = np.random.choice(non_full_indices)
                values[chosen_idx] += 1
                remaining -= 1

            return values

        target_pruned_channel = int(np.round(self.original_channels * self.percentage))
        target_block_pruned_channel = target_pruned_channel - self.dim_per_block[0] + np.random.randint(0, self.dim_per_block[0] + 1)

        upper_bounds = np.round(self.dim_per_block * 0.8).astype(int)[1:]
        pruned_channel_per_block = generate_random_values_with_limits(target_block_pruned_channel, upper_bounds)
        pruned_channel_conv1 = target_pruned_channel - np.sum(pruned_channel_per_block)
        candidate = np.concatenate(([pruned_channel_conv1], pruned_channel_per_block), axis=0)

        candidate = candidate / self.dim_per_block

        return candidate


    def engine(self, max_iters=100, crossover_num=25, mutation_num=25, prob=0.25, top_k=50):
        epoch = 0
        self.protect()
        while epoch < self.epochs:
            # Init self.candidates
            while len(self.candidates) < self.population:
                cand = self.get_random_candidates()
                self.candidates.add(tuple(cand))

            corssover_kids = self.crossover(crossover_num, max_iters)
            mutation_kids = self.mutation(mutation_num, max_iters, prob)

            self.candidates = self.candidates | corssover_kids | mutation_kids

            self.select(top_k=top_k)

            print(self.best)

            epoch += 1

        self.generate_masks()


    def protect(self):
        for index, (key, value) in enumerate(self.scores.items()):
            protect_rate = self.protect_percent
            for idx, arr in enumerate(value):
                if idx == len(value) - 1:
                    current_protect_rate = 0.5 * protect_rate
                else:
                    current_protect_rate = protect_rate
                protect_num = np.round(arr.shape[0] * current_protect_rate).astype(int)
                indices = np.argpartition(-arr, protect_num)[:protect_num]
                arr[indices] = 1e9

        return self.scores

    def generate_masks(self):
        pruning_rate = np.asarray(self.best)
        self.masks = list()

        for index, (key, value) in enumerate(self.scores.items()):
            saved_channels = self.dim_per_block[index] - int(np.round(pruning_rate[index] * self.dim_per_block[index]))
            flatten_score = np.concatenate(value)

            _, indices = torch.topk(torch.tensor(flatten_score), saved_channels, largest=True, sorted=False)
            mask = torch.zeros_like(torch.tensor(flatten_score), dtype=torch.bool)
            mask[indices] = 1
            mask = mask.numpy()

            original_shapes = [arr.shape for arr in value]  # 每个数组的形状
            split_indices = np.cumsum([arr.size for arr in value[:-1]])  # 分割点索引
            split_masks = np.split(mask, split_indices)  # 分割掩码
            reshaped_masks = [m.reshape(s) for m, s in zip(split_masks, original_shapes)]

            self.masks.append(reshaped_masks)

        if len(self.masks) == 17:
            groups = [3, 4, 6, 3]
            start_block = 1
            for group_size in groups:
                blocks = [i for i in range(start_block, start_block + group_size)]
                masks = np.asarray([self.masks[block][-1] for block in blocks])
                votes = np.sum(masks, axis=0)

                threshold = int(len(masks) / 2)
                result = votes >= threshold

                for block in blocks:
                    self.masks[block][-1] = result

                start_block += group_size


        elif len(self.masks) == 9:
            groups = [3, 2, 2, 2]
            start_block = 0
            for group_size in groups:
                blocks = [i for i in range(start_block, start_block + group_size)]
                masks = np.asarray([self.masks[block][-1] for block in blocks])
                votes = np.sum(masks, axis=0)

                result = votes >= 1

                for block in blocks:
                    self.masks[block][-1] = result

                start_block += group_size


    def get_mask(self):
        return self.masks

class Controller_vgg(object):
    def __init__(self,
                 model: timm.models.VGG,
                 scores,
                 pruning_rate,
                 population=50,
                 epsilon=0.02,
                 protect=0.0,
                 epochs=50):

        self.model = model
        self.scores = scores
        self.population = population


        self.percentage = pruning_rate
        self.epsilon = epsilon
        self.protect_percent = protect


        self.masks = None

    def protect(self):
        for key, value in self.scores.items():
            for idx, arr in enumerate(value):
                protect_num = np.round(arr.shape[0] * self.protect_percent).astype(int)
                indices = np.argpartition(-arr, protect_num)[:protect_num]
                arr[indices] = np.inf

    def engine(self, max_iters=100, crossover_num=25, mutation_num=25, prob=0.25, top_k=50 ):
        self.protect()


        pruned_channels = 0
        for key, value in self.scores.items():
            for idx, arr in enumerate(value):
                pruned_channels += np.round(len(arr) * self.percentage).astype(int)


        flattened_array = np.sort(np.concatenate(
            [item.flatten() if isinstance(item, np.ndarray) else np.array([item]) for item in self.scores['conv']]))

        self.masks = dict()

        threshold = flattened_array[pruned_channels]
        for key, value in self.scores.items():
            self.masks[key] = list()
            for idx, arr in enumerate(value):
                mask = arr >= threshold
                self.masks[key].append(mask)

    def get_mask(self):
        return self.masks















