def if_use_feat(caption_model):
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_fc, use_att = True, False
    elif caption_model == 'language_model':
        use_fc, use_att = False, False
    elif caption_model in ['topdown', 'aoa']:
        use_fc, use_att = True, True
    else:
        use_fc, use_att = False, True
    return use_fc, use_att