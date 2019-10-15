import pandas as pd
import datetime
import re
from sklearn.preprocessing import OneHotEncoder


def impute_missing_values(file):
    # Fill in the realise date and run time
    file['release_date'] = file['release_date'].fillna(method='ffill')
    file['runtime'] = file['runtime'].fillna(method='ffill')

    # Use the title to substitute the missing value of overview and tagline
    file['overview'].loc[file['overview'].isna()] = file['title']
    file['tagline'].loc[file['tagline'].isna()] = file['title']
    return file


def date_manipulation(file):
    file['year'] = file['month'] = file['day'] = 0
    for i in file['release_date']:
        file['year'].loc[file['release_date'] == i] = datetime.datetime.strptime(i, '%d-%m-%y').year
        file['month'].loc[file['release_date'] == i] = datetime.datetime.strptime(i, '%d-%m-%y').month
        file['day'].loc[file['release_date'] == i] = datetime.datetime.strptime(i, '%d-%m-%y').day
        return file


def one_hot_encoding(file):
    enc = OneHotEncoder()

    ol_mapping = {'en': 1, 'ja': 2, 'fr': 3, 'zh': 4, 'es': 5, 'de': 6, 'hi': 7, 'ru': 8, 'ko': 9, 'te': 10,
                  'cn': 11,
                  'it': 12, 'nl': 13, 'ta': 14, 'sv': 15, 'th': 16, 'da': 17, 'xx': 18, 'hu': 19, 'cs': 20,
                  'pt': 21, 'is': 22,
                  'tr': 23, 'nb': 24, 'af': 25, 'pl': 26, 'he': 27, 'ar': 28, 'vi': 29, 'ky': 30, 'id': 31,
                  'ro': 32, 'fa': 33,
                  'no': 34, 'sl': 35, 'ps': 36, 'el': 37}
    file['original_language'] = file['original_language'].map(ol_mapping)

    status_mapping = {'Released': 1, 'Post Production': 2, 'Rumored': 3}
    file['status'] = file['status'].map(status_mapping)

    onehot_features = ['original_language', 'status']
    enc.fit(file[onehot_features])

    enc_res = enc.transform(file[onehot_features])
    file = pd.concat([file, pd.DataFrame(enc_res.toarray())], axis=1)
    return file


'''This function mainly handles the columns which contain
more than one piece of information in every cell.single.
'''


def manual_one_hot_encoding(file):
    # genres
    file['genres_index'] = '0'
    for i in file.genres:
        if i != '[]':
            a = i.split(', {')
            for k in range(1, len(a)):
                a[k] = '{' + a[k]
            a[0] = a[0][1:]
            a[len(a) - 1] = a[len(a) - 1][:-1]
            b = ['' for i in range(len(a))]
            for q in range(len(a)):
                b[q] = int(''.join(re.findall(r'"id": ([0-9]*), "name"', a[q])))
            b = ','.join(str(m) for m in b)
            file['genres_index'].loc[file['genres'] == i] = b

    genres_list = []
    for i in file.genres_index:
        a = i.split(',')
        for j in range(len(a)):
            if int(a[j]) not in genres_list and a != '0':
                genres_list.append(int(a[j]))

    for i in range(len(genres_list)):
        new_label = 'genre_' + str(i + 1)
        file[new_label] = 0

    for i in file.genres_index:
        a = i.split(',')
        for j in a:
            for k in range(len(genres_list)):
                if int(j) == genres_list[k] and int(j) != 0:
                    file['genre_' + str(k + 1)].loc[file['genres_index'] == i] = 1

    # keywords
    file['keywords_index'] = '0'
    for i in file.keywords:
        if i != '[]':
            a = i.split(', {')
            for k in range(1, len(a)):
                a[k] = '{' + a[k]
            a[0] = a[0][1:]
            a[len(a) - 1] = a[len(a) - 1][:-1]
            b = ['' for i in range(len(a))]
            for q in range(len(a)):
                b[q] = int(''.join(re.findall(r'"id": ([0-9]*), "name"', a[q])))
            b = ','.join(str(m) for m in b)
            file['keywords_index'].loc[file['keywords'] == i] = b

    keywords_list = []
    for i in file.keywords_index:
        a = i.split(',')
        for j in range(len(a)):
            if int(a[j]) not in keywords_list and a[j] != '0':
                keywords_list.append(int(a[j]))

    for i in range(len(keywords_list)):
        file['keywords_' + str(i + 1)] = 0

    for i in file.keywords_index:
        a = i.split(',')
        for j in a:
            for k in range(len(keywords_list)):
                if int(j) == keywords_list[k] and j != '0':
                    file['keywords_' + str(k + 1)].loc[file['keywords_index'] == i] = 1

    # production companies
    file['production_companies_index'] = '0'
    for i in file.production_companies:
        if i != '[]':
            a = i.split(', {')
            for k in range(1, len(a)):
                a[k] = '{' + a[k]
            a[0] = a[0][1:]
            a[len(a) - 1] = a[len(a) - 1][:-1]
            b = ['' for i in range(len(a))]
            for q in range(len(a)):
                b[q] = int(''.join(re.findall(r'"id": ([0-9]*)}', a[q])))
            b = ','.join(str(m) for m in b)
            file['production_companies_index'].loc[file['production_companies'] == i] = b

    production_companies_list = []
    for i in file.production_companies_index:
        a = i.split(',')
        for j in range(len(a)):
            if int(a[j]) not in production_companies_list and a[j] != '0':
                production_companies_list.append(int(a[j]))

    for i in range(len(production_companies_list)):
        file['production_companies_' + str(i + 1)] = 0

    for i in file.production_companies_index:
        a = i.split(',')
        for j in a:
            for k in range(len(production_companies_list)):
                if int(j) == production_companies_list[k] and j != '0':
                    file['production_companies_' + str(k + 1)].loc[file['production_companies_index'] == i] = 1

    # production countries
    file['production_countries_index'] = '0'
    for i in file.production_countries:
        if i != '[]':
            a = i.split(', {')
            for k in range(1, len(a)):
                a[k] = '{' + a[k]
            a[0] = a[0][1:]
            a[len(a) - 1] = a[len(a) - 1][:-1]
            b = ['' for i in range(len(a))]
            for q in range(len(a)):
                b[q] = ''.join(re.findall(r'"iso_3166_1": "(.*)", "', a[q]))
            b = ','.join(str(m) for m in b)
            file['production_countries_index'].loc[file['production_countries'] == i] = b

    production_countries_list = []
    for i in file.production_countries_index:
        a = i.split(',')
        for j in range(len(a)):
            if a[j] not in production_countries_list and a[j] != '0':
                production_countries_list.append(a[j])

    for i in range(len(production_countries_list)):
        file['production_countries_' + str(1 + i)] = 0

    for i in file.production_countries_index:
        a = i.split(',')
        for j in a:
            for k in range(len(production_countries_list)):
                if j == production_countries_list[k] and j != '0':
                    file['production_countries_' + str(k + 1)].loc[file['production_countries_index'] == i] = 1

    # spoken languages
    file['spoken_languages_index'] = '0'
    for i in file.spoken_languages:
        if i != '[]':
            a = i.split(', {')
            for k in range(1, len(a)):
                a[k] = '{' + a[k]
            a[0] = a[0][1:]
            a[len(a) - 1] = a[len(a) - 1][:-1]
            b = ['' for i in range(len(a))]
            for q in range(len(a)):
                b[q] = ''.join(re.findall(r'"iso_639_1": "(.*)", "', a[q]))
            b = ','.join(str(m) for m in b)
            file['spoken_languages_index'].loc[file['spoken_languages'] == i] = b

    spoken_languages_list = []
    for i in file.spoken_languages_index:
        a = i.split(',')
        for j in range(len(a)):
            if a[j] not in spoken_languages_list and a[j] != '0':
                spoken_languages_list.append(a[j])

    for i in range(len(spoken_languages_list)):
        file['spoken_languages_' + str(1 + i)] = 0

    for i in file.spoken_languages_index:
        a = i.split(',')
        for j in a:
            for k in range(len(spoken_languages_list)):
                if j == spoken_languages_list[k] and j != '0':
                    file['spoken_languages_' + str(k + 1)].loc[file['spoken_languages_index'] == i] = 1

    return file


'''Drop the useless data columns'''


def drop_cols(file):
    file = file.drop(['genres', 'keywords', 'original_language', 'overview',
                    'production_companies', 'production_countries', 'release_date', 'spoken_languages',
                    'tagline', 'title', 'genre_index', 'genre_21', 'production_countries_index',
                    'production_companies_index', 'keywords_index', 'production_countries_index',
                    'spoken_languages_index', 'homepage', 'original_title'], axis=1)
    return file


def normalize(file):
    result = file.copy()
    normalized_features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    for f in normalized_features:
        max_ = file[f].max()
        min_ = file[f].min()
        result[f] = (file[f] - min_) / (max_ - min_)
    return result

    return file


# f = pd.read_csv('H:\\Downloads\\movies.csv')
# f = normalize(drop_cols(manual_one_hot_encoding(one_hot_encoding(date_manipulation(impute_missing_values(f))))))
# print(f.head(10))
