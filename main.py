#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code  performs deduplication of records with a comma separated values
(CSV) file.

We start with a CSV file containing listings of patient records.

The output will be a CSV with our clustered results.

"""
from future.builtins import next

import os
import csv
import re
import logging
import optparse

import dedupe
from unidecode import unidecode

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. This
# code block lets you change the level of loggin on the command
# line. You don't need it if you don't want that. To enable verbose
# logging, run `assignment/main.py -v`
optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose:
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)

# ## Setup

input_file = 'Deduplication Problem - Sample Dataset.csv'
output_file = 'Deduplication output.csv'
settings_file = 'settings'
training_file = 'training.json'

def preProcess(column):
    """
    Done a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines are ignored.
    """
    try : # python 2/3 string differences
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if not column:
        column = None
    return column

def readData(filename):
    """
    Read in the data from a CSV file and created a dictionary of records, 
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename, 'rU') as f:
        reader = csv.DictReader(f)
        ide = 0
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = ide
            data_d[row_id] = dict(clean_row)
            ide= ide + 1

    return data_d

print('importing data ...')
data_d = readData(input_file)

# If a settings file already exists, we'll just load that and skip training
if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f)
else:
    # ## Training

    # Define the fields it will pay attention to
    fields = [
        {'field' : 'ln', 'type': 'String'},
        {'field' : 'dob', 'type': 'Exact'},
        #{'field' : 'dob', 'type': 'String'},
        {'field' : 'gn', 'type': 'Exact'},
        {'field' : 'fn', 'type': 'String'},
        ]

    # Created a new object and passed our data model to it.
    deduper = dedupe.Dedupe(fields)

    # For training, I feed it a sample of records.
    deduper.sample(data_d)

      # If we have training data saved from a previous run,
  # look for it and load it in.
    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.readTraining(f)

    # ## Active learning
    # will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.
    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print('starting active labeling...')

    dedupe.consoleLabel(deduper)

    # Using the examples we just labeled, train it and learn
    # blocking predicates
    deduper.train()

    # save our training to disk
    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)
        
# Find the threshold that will maximize a weighted average of our
# precision and recall.  When we set the recall weight to 2, we are
# saying we care twice as much about recall as we do precision.
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.threshold(data_d, recall_weight=1)

# ## Clustering

# `match` will return sets of record IDs that it
# believes are all referring to the same entity.

print('clustering...')
clustered_dupes = deduper.match(data_d, threshold)

print('# duplicate sets', len(clustered_dupes))

# ## Writing Results

# Write our original data back out to a CSV with a new column called 
# 'Cluster ID' which indicates which records refer to each other.

cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id" : cluster_id,
            "canonical representation" : canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1

with open(output_file, 'w') as f_output, open(input_file , 'rU') as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)
    ide = 0
    for row in reader:
        row_id = ide
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, None)
            row.insert(0, singleton_id)
            singleton_id += 1
            for key in canonical_keys:
                row.append(None)
        writer.writerow(row)
        ide= ide + 1
