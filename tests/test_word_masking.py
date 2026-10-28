import numpy as np
import pytest
from scripts.word_masking_study import deleted_texts, expected_response


def test_word_deletion_changes_related_bigrams_after_revectorization():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer=TfidfVectorizer(ngram_range=(1,2)).fit(['good acting moving story'])
    variants=deleted_texts('good acting moving story',['good','acting','moving','story'])
    assert variants[15]=='good acting moving story'
    assert 'good' not in variants[14].split()
    x=vectorizer.transform(variants)
    bigram=vectorizer.vocabulary_['good acting']
    assert x[15,bigram]>0 and x[14,bigram]==0


def test_retention_expectation_has_correct_corner_and_midpoint_values():
    probabilities=np.arange(16,dtype=float)/16
    values=expected_response(probabilities,np.array([[0]*4,[1]*4,[.5]*4]))
    assert np.allclose(values,[probabilities[0],probabilities[-1],probabilities.mean()])


def test_quadratic_word_explanation_preserves_unmodeled_higher_order_error():
    from xgb_doe_xai.local_explanations import explain_word_response
    values=np.array([.2+.4*all(mask&(1<<i) for i in range(3)) for mask in range(16)])
    report=explain_word_response(lambda p:expected_response(values,p),['a','b','c','d'])
    assert report['confirmation']['rmse']>.001


def test_nested_registry_named_file_is_not_silently_ignored(tmp_path):
    from scripts.word_masking_study import assert_primary_registry
    (tmp_path/'programme_registry.json').write_text('{}')
    (tmp_path/'extra').mkdir()
    (tmp_path/'extra/programme_registry.json').write_text('{}')
    with pytest.raises(ValueError):
        assert_primary_registry(tmp_path)


def test_word_masking_roundtrip_uses_saved_classifier_and_literal_text_variants(tmp_path,monkeypatch):
    import csv,json
    from dataclasses import asdict
    from scripts import word_masking_study as study
    from xgb_doe_xai import generate_synthetic_sentiment_corpus,create_strictly_isolated_partitions,BenchmarkConfig
    from xgb_doe_xai.word_study import run_word_study
    from xgb_doe_xai.study import write_json
    campaign=tmp_path/'primary';campaign.mkdir()
    cfg=BenchmarkConfig(n_trees=2,n_nuisance_blocks=2,max_features=100)
    texts,labels=generate_synthetic_sentiment_corpus(100,42)
    dataset=tmp_path/'synthetic.csv'
    with dataset.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['text','label'])
        writer.writerows(zip(texts,labels))
    parts=create_strictly_isolated_partitions(texts,labels,n_blocks=2,max_features=100)
    contract=dict(configs={},dataset_paths={},dataset_sha256={})
    for name in ('imdb','yelp'):
        run_word_study(parts,cfg,{'coded_params':dict.fromkeys(['x1','x2','x3','x4'],0.)},
                       campaign/(name+'_words'),documents=2)
        contract['configs'][name]=asdict(cfg)
        contract['dataset_paths'][name]=str(dataset)
        contract['dataset_sha256'][name]=study.sha(dataset)
    write_json(campaign/'programme_contract.json',contract)
    write_json(campaign/'programme_registry.json',{p.relative_to(campaign).as_posix():study.sha(p)
                for p in campaign.rglob('*') if p.is_file()})
    protocol=tmp_path/'protocol.json';write_json(protocol,study.declared_protocol())
    monkeypatch.setattr(study,'check_study',lambda path:None)  # Primary fitting is covered separately.
    output=tmp_path/'masking'
    study.run(campaign,output,protocol)
    study.check(campaign,output)
    record=json.loads((output/'imdb_observations.json').read_text())
    assert len(record['documents'])==2
    assert len(record['documents'][0]['coalition_probabilities'])==16
