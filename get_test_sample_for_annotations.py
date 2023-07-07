from operator import itemgetter
import json
import random
import argparse
random.seed(42)



def create_sample_from_testfile(test_file):
    """read test file, get a random sample of 100, return the sample"""
    
    data_list = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            instance_dict = json.loads(line)
            data_list.append(instance_dict)
            
    print(f"Total number of test instances: {len(data_list)}")
    
    sample_list = random.sample(data_list, 100)
    
    print(f"Number of sample instances: {len(sample_list)}")
    
    sample_list = sorted(sample_list, key=itemgetter('id'))
    
    return sample_list

def write_the_sample(sample_list, outfile):
    # Let's save the data in a jsonl file
    with open(outfile, 'w', encoding='utf-8') as f:
        for line in sample_list:
            line = json.dump(line, f, ensure_ascii=False)
            f.write(f'{line}\n')
            
            
def main():
    print("Creating sample from test file...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--testfile", type=str, default="/home/finapolat/GenIE/data/rebel/en_test.jsonl")
    parser.add_argument("--outfolder", type=str, default="/home/finapolat/KGC-LLM/sample_from_testdata_for_annotations.jsonl")
    args = parser.parse_args()
    sample_list = create_sample_from_testfile(test_file=args.testfile)
    write_the_sample(sample_list, args.outfolder)


if __name__ == "__main__":
    print("Running get_test_sample_for_annotations.py...")
    main()