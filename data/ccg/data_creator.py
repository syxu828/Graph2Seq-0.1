import codecs
import json

def create(data_path, output_path):
    with codecs.open(data_path, 'r', "utf-8") as fr, codecs.open(output_path, 'w', 'utf-8') as fw:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split(" ")
            g_ids_features = {}
            g_adj ={}
            tags = []
            prev_id = -1
            for sub_info in info:
                temp = sub_info.split("|")
                id = len(g_ids_features)
                g_adj[id] = []
                g_ids_features[id] = temp[0]
                if prev_id != -1:
                    g_adj[prev_id].append(id)
                tags.append(temp[2])
                prev_id = id
            seq = " ".join(tags)

            jo = {"g_ids_features":g_ids_features, "seq":seq, "g_adj":g_adj}
            fw.write(json.dumps(jo)+"\n")

if __name__ == "__main__":
    create("train.stagged", "train.data")
    create("dev.stagged", "dev.data")
    create("test.stagged", "test.data")