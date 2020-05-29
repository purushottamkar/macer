import pickle
with open('macer/edit_pos_tokenizer','rb') as file:
    edit_tkn=pickle.load(file)
with open('macer/feature_tokenizer','rb') as file:
    tkn=pickle.load(file)
with open('macer/idx_to_bigram','rb') as file:
    idx_to_bigram=pickle.load(file)
with open('macer/new_encoder','rb') as file:
    new_encoder=pickle.load(file)
with open('macer/flat_rankers','rb') as file:
    flat_rankers=pickle.load(file)
with open('macer/repl_or_not','rb') as file:
    repl_or_not=pickle.load(file)
with open('macer/ins_del_model','rb') as file:
    ins_del_model=pickle.load(file)
with open('macer/repl_class_model','rb') as file:
    repl_class_model=pickle.load(file)
with open('macer/repl_encoder','rb') as file:
    repl_encoder=pickle.load(file)
with open('macer/ins_class_model','rb') as file:
    ins_class_model=pickle.load(file)
with open('macer/ins_encoder','rb') as file:
    ins_encoder=pickle.load(file)
with open('macer/del_class_model','rb') as file:
    del_class_model=pickle.load(file)
with open('macer/del_encoder','rb') as file:
    del_encoder=pickle.load(file)
with open('macer/rest_class_model','rb') as file:
    rest_class_model=pickle.load(file)
with open('macer/rest_encoder','rb') as file:
    rest_encoder=pickle.load(file)
with open('macer/repl_clusters','rb') as file:
    repl_clusters=pickle.load(file)
with open('macer/ins_clusters','rb') as file:
    ins_clusters=pickle.load(file)
with open('macer/del_clusters','rb') as file:
    del_clusters=pickle.load(file)
with open('macer/rest_clusters','rb') as file:
    rest_clusters=pickle.load(file)

corr_cls=0
rerank=0
bigram_rank=0
fixer=0
