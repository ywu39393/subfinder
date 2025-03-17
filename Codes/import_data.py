import os
import sys
from collections import defaultdict

def read_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except IOError:
        print(f'Failed to open or read the file: {file_path}')
        sys.exit(-1)
    
    return [line.strip() for line in lines if line.strip()]

def parse_line(line):
    parts = line.split('\t')
    cgc_id = f"{parts[2]}|{parts[0]}"
    gene_type = parts[1]
    annotation = parts[7].replace(' ', '')  # Remove any spaces from annotation
    
    if gene_type == 'null':
        annotation = 'null'
    elif gene_type == 'TC':
        annotation = '.'.join(annotation.split('.')[:3])
    elif gene_type in ['TF', 'STP']:
        annotation = annotation.replace('+', '|')
    # For CAZyme and other cases, keep annotation as is
    
    return cgc_id, annotation

def reformat_cgc(input_file, output_file):
    lines = read_file(input_file)
    cgc_dict = defaultdict(list)
    
    for line in lines[1:]:  # Skip header
        cgc_id, annotation = parse_line(line)
        cgc_dict[cgc_id].append(annotation)
    
    with open(output_file, 'w') as f:
        for cgc_id, annotations in cgc_dict.items():
            f.write(f"{cgc_id}\t{','.join(annotations)}\n")