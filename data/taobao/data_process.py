import pandas as pd
import numpy as np
import random

# data = pd.read_csv("raw_sample.csv")
#
# is_clk = data.loc[:, 'clk']
#
# data_preprocessed = data.loc[is_clk == 1, ['user', 'adgroup_id', 'pid']]
# data_preprocessed.to_csv("preprocessed.csv", index=False)

# user_data = pd.read_csv("user_profile.csv")
# user_set = set()
# for i in range(len(user_data)):
#     user_set.add(user_data.iloc[i, 0])
# degree = 5
# data = pd.read_csv("preprocessed.csv")
# user_dic = {}
# rest_dic_f = {}
# user_dic_f = {}
# rest_set = set()
#
# total_lines = len(data)
#
# for line in range(total_lines):
#     if data.iloc[line, 0] not in user_dic_f.keys() and data.iloc[line, 0] in user_set:
#         user_dic_f[data.iloc[line, 0]] = set()
#         user_dic_f[data.iloc[line, 0]].add(data.iloc[line, 1])
#     elif data.iloc[line, 0] in user_set:
#         user_dic_f[data.iloc[line, 0]].add(data.iloc[line, 1])
#
# for k in user_dic_f.keys():
#     if len(user_dic_f[k]) >= degree:
#         user_dic[k] = len(user_dic)
#         for l in user_dic_f[k]:
#             if l not in rest_dic_f.keys():
#                 rest_dic_f[l] = 0
#             rest_dic_f[l] = rest_dic_f[l] + 1
#             if rest_dic_f[l] == degree:
#                 rest_set.add(l)
#
# flag = True
# max_user = 0
# while flag:
#     max_user = 0
#     flag = False
#     now = len(user_dic)
#     user_dic = {}
#     for i in rest_dic_f.keys():
#         rest_dic_f[i] = 0
#     for k in user_dic_f.keys():
#         user_dic_f[k] = user_dic_f[k].intersection(rest_set)
#         if len(user_dic_f[k]) >= degree:
#             max_user = max(max_user, k)
#             user_dic[k] = len(user_dic)
#             for l in user_dic_f[k]:
#                 rest_dic_f[l] = rest_dic_f[l] + 1
#                 if rest_dic_f[l] == degree:
#                     rest_set.add(l)
#     if now != len(user_dic):
#         flag = True
#
# print(len(rest_set))
# print(len(user_dic))
#
# cms_segid_dic = {}
# cms_group_id_dic = {}
# user_data = pd.read_csv("user_profile.csv")
# max_arg = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# total = 0
# with open("temp.user", 'w+') as write_file:
#     for index in range(len(user_data)):
#         if user_data.iloc[index, 0] in user_dic.keys():
#             total += 1
#             write_line = str(user_dic[user_data.iloc[index, 0]]) + " "
#             if user_data.iloc[index, 1] not in cms_segid_dic.keys():
#                 cms_segid_dic[user_data.iloc[index, 1]] = len(cms_segid_dic)
#             if user_data.iloc[index, 2] not in cms_group_id_dic.keys():
#                 cms_group_id_dic[user_data.iloc[index, 2]] = len(cms_group_id_dic)
#             write_line += str(user_data.iloc[index, 1]) + " " + str(user_data.iloc[index, 2]) + " "
#             for j in range(3, 9):
#                 if j == 5 or j == 8:
#                     if np.isnan(user_data.iloc[index, j]):
#                         user_data.iloc[index, j] = 0
#                     else:
#                         user_data.iloc[index, j] = int(user_data.iloc[index, j])
#                 write_line += str(int(user_data.iloc[index, j])) + " "
#             for j in range(0, 9):
#                 max_arg[j] = max(max_arg[j], user_data.iloc[index, j])
#             write_line += '\n'
#             write_file.write(write_line)
# print(max_arg)
# print(total)
#
# item_dic = {}
# for item in rest_set:
#     item_dic[item] = len(item_dic)
# item_max_arg = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
# item_data = pd.read_csv("ad_feature.csv")
# cate_id_dict = {}
# campaign_id_dict = {}
# customer_id_dic = {}
# brand_id_dic = {}
# price_dic = {0: 0, 1: 0, 2: 0, 3: 0}
#
#
# def price_level_split(price):
#     my_price = int(price)
#     if my_price < 100:
#         return 0
#     else:
#         return 1
#
#
# with open("temp.item", 'w+') as write_file:
#     for index in range(len(item_data)):
#         if item_data.iloc[index, 0] in item_dic.keys():
#             write_line = str(item_dic[item_data.iloc[index, 0]]) + " "
#             item_max_arg[0] = max(item_max_arg[0], item_dic[item_data.iloc[index, 0]])
#             if item_data.iloc[index, 1] not in cate_id_dict.keys():
#                 cate_id_dict[item_data.iloc[index, 1]] = len(cate_id_dict)
#             write_line += str(cate_id_dict[item_data.iloc[index, 1]]) + " "
#             item_max_arg[1] = max(item_max_arg[1], cate_id_dict[item_data.iloc[index, 1]])
#             # if item_data.iloc[index, 2] not in campaign_id_dict.keys():
#             #     campaign_id_dict[item_data.iloc[index, 2]] = len(campaign_id_dict)
#             # write_line += str(campaign_id_dict[item_data.iloc[index, 2]]) + " "
#             # item_max_arg[2] = max(item_max_arg[2], campaign_id_dict[item_data.iloc[index, 2]])
#             # if item_data.iloc[index, 3] not in customer_id_dic.keys():
#             #     customer_id_dic[item_data.iloc[index, 3]] = len(customer_id_dic)
#             # write_line += str(customer_id_dic[item_data.iloc[index, 3]]) + " "
#             # item_max_arg[3] = max(item_max_arg[3], customer_id_dic[item_data.iloc[index, 3]])
#             if np.isnan(item_data.iloc[index, 4]):
#                 item_data.iloc[index, 4] = 0
#             if item_data.iloc[index, 4] not in brand_id_dic.keys():
#                 brand_id_dic[item_data.iloc[index, 4]] = len(brand_id_dic)
#             write_line += str(brand_id_dic[item_data.iloc[index, 4]]) + " "
#             item_max_arg[4] = max(item_max_arg[4], brand_id_dic[item_data.iloc[index, 4]])
#             price_level = price_level_split(item_data.iloc[index, 5])
#             write_line += str(price_level)
#             write_line += '\n'
#             write_file.write(write_line)
#     print(item_max_arg)
#
# with open('train.txt', 'w+') as train_file:
#     with open('test.txt', 'w+') as test_file:
#         for k in user_dic.keys():
#             item_list = list(user_dic_f[k])
#             random.shuffle(item_list)
#             test_len = int(round(len(item_list) / 5))
#             train_list = item_list[0:-test_len]
#             test_list = item_list[-test_len:]
#             train_str = str(user_dic[k])
#             test_str = str(user_dic[k])
#             for i in train_list:
#                 train_str += " " + str(item_dic[i])
#             for i in test_list:
#                 test_str += " " + str(item_dic[i])
#             train_str += '\n'
#             test_str += '\n'
#             train_file.write(train_str)
#             test_file.write(test_str)

u_item_dic = {}
u_user_dic = {}

with open("temp.item",'r',encoding="ISO-8859-1") as read_file:
    for line in read_file.readlines():
        if len(line.split(" ")) != 4:
            print("error1")
        l = int((line.split(" "))[0])
        u_item_dic[l] = line
    with open("u.item",'w+') as write_file:
        for i in range(len(u_item_dic)):
            write_file.write(u_item_dic[i])

with open("temp.user",'r',encoding="ISO-8859-1") as read_file:
    for line in read_file.readlines():
        if len(line.split(" ")) != 10:
            print("error2")
        l = int((line.split(" "))[0])
        u_user_dic[l] = line
    with open("u.user",'w+') as write_file:
        for i in range(len(u_user_dic)):
            write_file.write(u_user_dic[i])
