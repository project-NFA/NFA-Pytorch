import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['ml-100k', 'ml-1m', 'yelp', 'taobao']:
    if world.model_name in['nfalgn', 'nfapinsage', 'nfangcf', 'nfagcn']:
        if world.dataset == 'taobao':
            user_index = [97,13,3,7,4,4,2,5]
            item_index = [1079,7565,2]
        if world.dataset =='ml-100k':
            user_index = [14, 2, 21]
            item_index = [3] + [2] * 19
        if world.dataset == 'yelp':
            user_index = [3,19,17,9,5]
            item_index = [836,31,8,2,2,2,2,2,2,4,2,3,2,2,2,3,2]
        if world.dataset =='ml-1m':
            user_index = [7, 2, 21]
            item_index = [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        dataset = dataloader.Loader(path="../data/" + world.dataset, required_feature=True, user_index=user_index, item_index=item_index)
    else:
        dataset = dataloader.Loader(path="../data/"+world.dataset)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'nfalgn' : model.NFALightGCN,
    'nfagcn' : model.NFAGCN,
    'nfapinsage' : model.NFAPinSage,
    'nfangcf' : model.NFANGCF,
}
