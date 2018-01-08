import os
import click
from json import dumps
from numpy import mean, nanmedian
from .. import load, score

@click.argument('files', nargs=2, metavar='<files: ground truth, estimate>', required=True)
@click.command('evaluate', short_help='compare two sets of results', options_metavar='<options>')
def evaluate(files):
    a = load(files[0])
    b = load(files[1])

    scores = {}

    for method in ['corr', 'rank', 'info', 'loglik']:
      allscores = score(a, b, method=method)
      scores[method] = nanmedian(allscores)

    print(dumps(scores))