import pandas as pd


def parse_libsvm_ranking(path, n_features):
    data = []
    query_ids = []
    targets = []

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            target = int(parts[0])
            qid = int(parts[1].split(':')[1])
            features = [0.0] * n_features

            for item in parts[2:]:
                if item.startswith('#'):
                    break  # комментарии игнорируем
                index, value = item.split(':')
                features[int(index) - 1] = float(value)

            data.append(features)
            query_ids.append(qid)
            targets.append(target)

    df = pd.DataFrame(data, columns=[f'f{i+1}' for i in range(n_features)])
    df['qid'] = query_ids
    df['target'] = targets
    return df

df = parse_libsvm_ranking("D:/ds/pets/mslr/data/Fold1/test.txt", n_features=136)

print(df.head())

df.to_csv(index = False, path_or_buf="D:/ds/pets/mslr/data/Fold1/mslrTest.csv")