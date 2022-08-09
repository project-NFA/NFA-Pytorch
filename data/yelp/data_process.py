import json
import random

# 160585 restaurants
# 19953 5-star restaurants
# 836 cities
# 31 states
# 25720 item 33078 user

city_dic = {}
state_dic = {}
rest_dic = dict()
user_dic_f = dict()
user_dic = dict()
rest_list = list()
rest_set_o = set()
user_list = []
rest_set = set()
rest_dic_f = {}
attribute_set = set()

# with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         data = json.loads(line)
#         if data["stars"] == 5.0:
#             if data["user_id"] not in user_dic_f.keys():
#                 user_dic_f[data["user_id"]] = set()
#                 user_dic_f[data["user_id"]].add(data["business_id"])
#             else:
#                 user_dic_f[data["user_id"]].add(data["business_id"])
#
# for k in user_dic_f.keys():
#     if len(user_dic_f[k]) >= 10:
#         user_dic[k] = len(user_dic)
#         for l in user_dic_f[k]:
#             if l not in rest_dic_f.keys():
#                 rest_dic_f[l] = 0
#             rest_dic_f[l] = rest_dic_f[l] + 1
#             if rest_dic_f[l] == 10:
#                 rest_set.add(l)
#
# flag = True
# while flag:
#     flag = False
#     now = len(user_dic)
#     user_dic = {}
#     for i in rest_dic_f.keys():
#         rest_dic_f[i] = 0
#     for k in user_dic_f.keys():
#         user_dic_f[k] = user_dic_f[k].intersection(rest_set)
#         if len(user_dic_f[k]) >= 10:
#             user_dic[k] = len(user_dic)
#             for l in user_dic_f[k]:
#                 rest_dic_f[l] = rest_dic_f[l] + 1
#                 if rest_dic_f[l] == 10:
#                     rest_set.add(l)
#     if now != len(user_dic):
#         flag = True
#
# for k in rest_set:
#     rest_dic[k] = len(rest_dic)
#
#
# def level_split(data_level):
#     if data_level < 60:
#         return data_level // 10
#     elif data_level < 120:
#         return 6 + (data_level - 60) // 20
#     elif data_level < 600:
#         return 9 + (data_level - 120) // 80
#     elif data_level < 2100:
#         return 15 + (data_level - 600) // 500
#     else:
#         return 18
#
#
# def review_level_split(data_level):
#     if data_level < 350:
#         return data_level // 50
#     else:
#         return 7


# attribute_dic = {'BusinessAcceptsCreditCards': 0, 'BusinessParking:valet': 1, 'BusinessParking:validated': 2,
#                  'BusinessParking:garage': 3, 'BusinessParking:lot': 4, 'BusinessParking:street': 5,
#                  'RestaurantsPriceRange2:int': 6, 'BikeParking': 7, 'WiFi:str': 8, 'GoodForKids': 9,
#                  'RestaurantsTakeOut': 10, 'Caters': 11, 'Alcohol:str': 12, 'RestaurantsDelivery': 13}
#
# attributes_num = {}
# attributes_category = {}
# review_dic = {}
#
# with open('yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
#     with open('u.item', 'w+') as write_file:
#         for line in f.readlines():
#             data = json.loads(line)
#             if data["business_id"] in rest_set:
#                 if data["city"] not in city_dic.keys():
#                     city_dic[data["city"]] = len(city_dic)
#                 if data["state"] not in state_dic.keys():
#                     state_dic[data["state"]] = len(state_dic)
#                 write_line = str(rest_dic[data["business_id"]]) + " " + str(city_dic[data["city"]]) \
#                              + " " + str(state_dic[data["state"]]) + " "
#                 review_level = review_level_split(data["review_count"])
#                 write_line = write_line + str(review_level) + " "
#                 attribute_line = [0 for i in range(14)]
#                 if type(data["attributes"]) == dict:
#                     for a in data["attributes"]:
#                         pre = a + ':'
#                         l = data["attributes"][a][1:-1].split(',')
#                         if len(l) >= 2:
#                             for index in range(len(l)):
#                                 if index:
#                                     t = ''.join(l[index])[2:].split('\'')
#                                 else:
#                                     t = ''.join(l[index])[1:].split('\'')
#                                 s = pre + t[0]
#                                 if t[1][2] in [str(i) for i in range(10)]:
#                                     s = s + ':int'
#                                     attribute_line[attribute_dic[s]] = int(t[1][2]) - 1
#                                 elif t[1][2] not in ['T', 'F']:
#                                     s = s + ':str'
#                                     if s in attribute_dic.keys():
#                                         if t[1][2] == 'f' or t[1][4] == 'f':
#                                             attribute_line[attribute_dic[s]] = 1
#                                         elif t[1][2] in ['p', 'b'] or t[1][4] in ['p', 'b']:
#                                             attribute_line[attribute_dic[s]] = 2
#                                 if t[1][2] != 'N':
#                                     if s in attribute_dic.keys() and t[1][2] == 'T':
#                                         attribute_line[attribute_dic[s]] = 1
#                                     # attribute_set.add(s)
#                                     # if s in attributes_num.keys():
#                                     #     attributes_num[s] = attributes_num[s] + 1
#                                     #     if t[1][2:] not in attributes_category[s]:
#                                     #         attributes_category[s].append(t[1][2:])
#                                     # else:
#                                     #     attributes_num[s] = 1
#                                     #     attributes_category[s] = [t[1][2:]]
#                         else:
#                             now = a
#                             if data["attributes"][a][0] in [str(i) for i in range(10)]:
#                                 now = now + ':int'
#                                 attribute_line[attribute_dic[now]] = int(data["attributes"][a][0]) - 1
#                             elif data["attributes"][a][0] not in ['T', 'F']:
#                                 now = now + ':str'
#                                 if now in attribute_dic.keys():
#                                     if data["attributes"][a][0] == 'f' or data["attributes"][a][2] == 'f':
#                                         attribute_line[attribute_dic[now]] = 1
#                                     elif data["attributes"][a][0] in ['p', 'b'] or data["attributes"][a][2] in ['p', 'b']:
#                                         attribute_line[attribute_dic[now]] = 2
#                             if data["attributes"][a][0] != 'N':
#                                 if now in attribute_dic.keys() and data["attributes"][a] == 'True':
#                                     attribute_line[attribute_dic[now]] = 1
#                                 # attribute_set.add(now)
#                                 # if now in attributes_num.keys():
#                                 #     attributes_num[now] = attributes_num[now] + 1
#                                 #     if data["attributes"][a] not in attributes_category[now]:
#                                 #         attributes_category[now].append(data["attributes"][a])
#                                 # else:
#                                 #     attributes_num[now] = 1
#                                 #     attributes_category[now] = [data["attributes"][a]]
#                 for i in attribute_line:
#                     write_line = write_line + str(i) + " "
#                 write_line += '\n'
#                 write_file.write(write_line)

# attribute_set = sorted(attribute_set)
# print(len(attribute_set))
# print(attribute_set)
# keys = [k for k, v in sorted(attributes_num.items(), key=lambda item: item[1], reverse=True)]
# for i in keys:
#     if len(attributes_category[i]) > 1:
#         attributes_category[i].append(str(attributes_num[i]))
#     else:
#         attributes_category.pop(i)
#     if i in attributes_category.keys() and 'False' in attributes_category[i]:
#         attributes_category[i] = [str(attributes_num[i])]
#
# attributes_category = {k: v for k, v in
#                        sorted(attributes_category.items(), key=lambda item: int(item[1][len(item[1]) - 1]),
#                               reverse=True)}


# print(len(attributes_category))
# print(attributes_category)


# def fans_level_split(data_level):
#     if data_level < 5:
#         return data_level
#     elif data_level < 20:
#         return 5 + (data_level - 5) // 5
#     else:
#         return 8


# star_dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
# review_dic = {}
# fans_dict = {}
# max_l = 0
# with open('yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
#     with open('u.user', 'w+') as write_file:
#         for line in f.readlines():
#             data = json.loads(line)
#             if data["user_id"] in user_dic.keys():
#                 write_line = str(user_dic[data["user_id"]]) + " "
#                 l = data["elite"].split(',')
#                 data["elite"] = len(l)
#                 write_line += str(2 if review_level_split(data["review_count"]) > 1
#                                   else review_level_split(data["review_count"])) + " "
#                 write_line += str(level_split(data["useful"])) + " "
#                 write_line += str(len(l)) + " "
#                 if len(l) == 16:
#                     print(user_dic[data["user_id"]])
#                 fans_level = fans_level_split(data["fans"])
#                 write_line += str(fans_level) + " "
#                 if data["average_stars"] < 3.5:
#                     star_level = 0
#                 else:
#                     star_level = (data["average_stars"] - 3) // 0.5
#                 write_line += str(int(star_level)) + '\n'
#                 if user_dic[data["user_id"]] == 5854:
#                     print(data)
#                     print(write_line)
#                 write_file.write(write_line)

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
#                 train_str += " " + str(rest_dic[i])
#             for i in test_list:
#                 test_str += " " + str(rest_dic[i])
#             train_str += '\n'
#             test_str += '\n'
#             train_file.write(train_str)
#             test_file.write(test_str)

u_item_dic = {}
u_user_dic = {}

with open("temp.item",'r',encoding="ISO-8859-1") as read_file:
    for line in read_file.readlines():
        # print(len(line.split(" ")))
        if len(line.split(" ")) != 19:
            print("error")
        l = int((line.split(" "))[0])
        u_item_dic[l] = line
    with open("u.item",'w+') as write_file:
        for i in range(len(u_item_dic)):
            write_file.write(u_item_dic[i])

with open("temp.user",'r',encoding="ISO-8859-1") as read_file:
    for line in read_file.readlines():
        l = int((line.split(" "))[0])
        u_user_dic[l] = line
    with open("u.user",'w+') as write_file:
        for i in range(len(u_user_dic)):
            write_file.write(u_user_dic[i])

