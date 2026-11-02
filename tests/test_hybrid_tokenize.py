# -*- coding: utf-8 -*-
"""tokenize 回归：二元组切分、脚本分区、NFKC 全角归一。"""

from app.retrieval.hybrid import tokenize


def test_cjk_bigram_sliding_window():
    assert tokenize("整车质保期是多久") == [
        "整车", "车质", "质保", "保期", "期是", "是多", "多久",
    ]


def test_latin_words_kept_whole():
    # 英文/数字串整词保留，不进二元组；小写归一
    assert tokenize("秦Plus DMI") == ["plus", "dmi", "秦"]


def test_punctuation_breaks_runs_and_is_dropped():
    # 标点断段且不进词项；二元组不跨标点
    assert tokenize("整车质保，电池保修") == ["整车", "车质", "质保", "电池", "池保", "保修"]


def test_nfkc_folds_fullwidth_alphanumerics():
    # 全角字母/数字若不归一会被两个字符类静默丢弃
    # （实现先收集英文/数字词再处理中文段，BM25 词袋不关心顺序）
    assert tokenize("质保６年") == ["6", "质保", "年"]
    assert tokenize("安全气囊（ＳＲＳ）展开") == [
        "srs", "安全", "全气", "气囊", "展开",
    ]
