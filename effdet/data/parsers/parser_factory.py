""" Parser factory

Copyright 2020 Ross Wightman
"""
from .parser_coco import CocoParser


def create_parser(name, **kwargs):
    if name == 'coco':
        parser = CocoParser(**kwargs)
    elif name == 'voc':
        raise NotImplementedError('COCO only for probability approach')
    elif name == 'openimages':
        raise NotImplementedError('COCO only for probability approach')
    else:
        assert False, f'Unknown dataset parser ({name})'
    return parser
