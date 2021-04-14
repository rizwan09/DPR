import json
from sklearn.model_selection import train_test_split

path='/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/c_functions_all_data.jsonl'
prject_a='c-project'
prject_b='homebrew'
data=[json.loads(line) for line in open(path)]
train_data = [ d for d in data if prject_a in d['file_path']]
out_of_domain_test_data = [ d for d in data if prject_a not in d['file_path'] and prject_b not in d['file_path']]
other_data = [ d for d in data if prject_b  in d['file_path'] ]

in_domain_valid_size = 4332
in_domain_test_size = 4203
in_domain_additional_train_size = len(other_data) - in_domain_test_size - in_domain_valid_size


X_train, X_dev_test = train_test_split( other_data, test_size=1-in_domain_additional_train_size/len(other_data), random_state=42)
in_domain_valid_data, in_domain_test_data = train_test_split( X_dev_test, test_size=in_domain_test_size/len(X_dev_test), random_state=42)

in_domain_train_data = train_data  + X_train


print("size: in_domain_train_data: ", len(in_domain_train_data))
print("size: in_domain_valid_data: ", len(in_domain_valid_data))
print("size: in_domain_test_data: ", len(in_domain_test_data))
print("size: out_of_domain_test_data: ", len(out_of_domain_test_data))

with open("in_domain_train.jsonl", 'w') as in_dom_train_file, \
        open("out_domain_test.jsonl", 'w') as out_dom_test_file, \
        open("in_domain_valid.jsonl", 'w') as in_dom_valid_file, \
        open("in_domain_test.jsonl", 'w') as in_dom_test_file, \
        open("overall_test.jsonl", 'w') as overall_test_file:



    for d in in_domain_train_data:
        in_dom_train_file.write(json.dumps(d)+"\n")

    for d in in_domain_valid_data:
        in_dom_valid_file.write(json.dumps(d) + "\n")

    for d in in_domain_test_data:
        in_dom_test_file.write(json.dumps(d) + "\n")
        overall_test_file.write(json.dumps(d) + "\n")

    for d in out_of_domain_test_data:
        out_dom_test_file.write(json.dumps(d) + "\n")
        overall_test_file.write(json.dumps(d) + "\n")






