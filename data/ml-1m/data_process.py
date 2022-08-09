import random

# 5652/6040 users
# 3230/3952 movies

user_list = [[i] for i in range(1, 6041)]

movies_set = set()
# 对rating过滤，只选5分的边
with open('ratings.dat', 'r', encoding="ISO-8859-1") as graph_file:
    for line in graph_file.readlines():
        l = line[0: -1].split('::')
        if l[2] == '5':
            user_list[int(l[0]) - 1].append(int(l[1]))
        # 将边处理成user item1 item2...的形式

user_dict = {}
movies_dict = {}
user_index = 0
movies_index = 0

# 对item节点进行过滤，只选取和user相连的item节点
# 对user进行过滤，只选度大于等于5的user节点
for l in user_list:
    if len(l) > 5:
        user_dict[l[0]] = user_index
        for v in l[1:]:
            movies_set.add(v)
        user_index += 1

for i in range(1, 3953):
    if i in movies_set:
        movies_dict[i] = movies_index
        movies_index += 1

print(len(user_dict))
print(len(movies_dict))

# 数据集
# 	  1:  "Under 18"
# 	 18:  "18-24"
# 	 25:  "25-34"
# 	 35:  "35-44"
# 	 45:  "45-49"
# 	 50:  "50-55"
# 	 56:  "56+"

# 0: "other" or not specified
# 1: "academic/educator"
# 2: "artist"
# 3: "clerical/admin"
# 4: "college/grad student"
# 5: "customer service"
# 6: "doctor/health care"
# 7: "executive/managerial"
# 8: "farmer"
# 9: "homemaker"
# 10: "K-12 student"
# 11: "lawyer"
# 12: "programmer"
# 13: "retired"
# 14: "sales/marketing"
# 15: "scientist"
# 16: "self-employed"
# 17: "technician/engineer"
# 18: "tradesman/craftsman"
# 19: "unemployed"
# 20: "writer"

age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

with open('users.dat', 'r', encoding="ISO-8859-1") as read_file:
    with open('users_refine.dat', 'w+') as write_file:
        for line in read_file.readlines():
            l = line[0: -1].split('::')
            if int(l[0]) in user_dict:
                gen = 0
                if l[1] == 'F':
                    gen = 1
                # index age gen occ

                write_line = str(user_dict[int(l[0])]) + ' ' + str(age_dict[int(l[2])]) + ' ' + str(gen) \
                             + ' ' + l[3] + '\n'
                write_file.write(write_line)

# 	 Action
# 	 Adventure
# 	 Animation
# 	 Children's
# 	 Comedy
# 	 Crime
# 	 Documentary
# 	 Drama
# 	 Fantasy
# 	 Film-Noir
# 	 Horror
# 	 Musical
# 	 Mystery
# 	 Romance
# 	 Sci-Fi
# 	 Thriller
# 	 War
# 	 Western

genre_dict = {'Action': 0, 'Adventure': 1, 'Animation': 2, 'Children\'s': 3, 'Comedy': 4, 'Crime': 5, 'Documentary': 6,
              'Drama': 7, 'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'Musical': 11, 'Mystery': 12, 'Romance': 13,
              'Sci-Fi': 14, 'Thriller': 15, 'War': 16, 'Western': 17}

with open('movies.dat', 'r', encoding="ISO-8859-1") as read_file:
    with open('movies_refine.dat', 'w+') as write_file:
        for line in read_file.readlines():
            l = line[0: -1].split('::')
            if int(l[0]) in movies_dict:
                # 对电影的发行年份做处理
                year = 0
                if int(l[1][-5:-1]) < 1960:
                    year = 2
                elif int(l[1][-5:-1]) < 1980:
                    year = 1
                # index year genre
                write_line = str(movies_dict[int(l[0])]) + ' ' + str(year)
                genre = l[2].split('|')
                movie_genre_dict = {}
                for v in range(18):
                    movie_genre_dict[v] = 0
                for g in genre:
                    movie_genre_dict[genre_dict[g]] = 1
                for v in range(18):
                    write_line += ' ' + str(movie_genre_dict[v])
                write_line += '\n'
                write_file.write(write_line)

with open('train.txt', 'w+') as train_file:
    with open('test.txt', 'w+') as test_file:
        for l in user_list:
            if len(l) > 5:
                movies_list = l[1:]
                random.shuffle(movies_list)

                test_len = int(round(len(l) / 5))

                train_list = movies_list[0:-test_len]
                test_list = movies_list[-test_len:]
                train_str = str(user_dict[l[0]])
                test_str = str(user_dict[l[0]])

                for v in train_list:
                    train_str += " " + str(movies_dict[v])
                for v in test_list:
                    test_str += " " + str(movies_dict[v])
                train_str += '\n'
                test_str += '\n'
                train_file.write(train_str)
                test_file.write(test_str)
