## Предобработка данных 
Удалил скрипты, которые превращали telegram комменты и youtube комменты в отформатированные csv файлы

## Классификация языка комментария
Использую библиотеку [fast-langdetect](https://github.com/LlmKira/fast-langdetect) для детекции.
### Файлы 
- classify_comment_languages.py - классификация обоих файлов 
- show_language_stats.py - визуализация распределения языков в сообщениях
**Результаты show_language_stats.py**:
```txt
LANGUAGE CLASSIFICATION RESULTS
==================================================

==================================================
YOUTUBE COMMENTS LANGUAGE DISTRIBUTION
==================================================
Total comments: 43,578
File size: 10.37 MB

Language Distribution:

Top 20 languages:
   1.     ru:   41,464 comments ( 95.15%)
   2.     uk:      675 comments (  1.55%)
   3.     en:      266 comments (  0.61%)
   4. unknown:      236 comments (  0.54%)
   5.     bg:      116 comments (  0.27%)
   6.     ja:       77 comments (  0.18%)
   7.     sr:       65 comments (  0.15%)
   8.     zh:       65 comments (  0.15%)
   9.     mk:       62 comments (  0.14%)
  10.     kk:       52 comments (  0.12%)
  11.     pl:       41 comments (  0.09%)
  12.     de:       35 comments (  0.08%)
  13.    ceb:       34 comments (  0.08%)
  14.     sl:       33 comments (  0.08%)
  15.     it:       28 comments (  0.06%)
  16.     be:       18 comments (  0.04%)
  17.     es:       18 comments (  0.04%)
  18.     fr:       18 comments (  0.04%)
  19.     ky:       16 comments (  0.04%)
  20.     eo:       15 comments (  0.03%)

Summary:
  Total unique languages detected: 67
  Most common language: ru (41,464 comments, 95.15%)

Language Detection Confidence:
  Mean confidence score: 0.956
  Median confidence score: 0.996
  Comments with high confidence (>0.9): 39,656 (91.00%)
  Comments with low confidence (<0.5): 1,205 (2.77%)

==================================================
TELEGRAM COMMENTS LANGUAGE DISTRIBUTION
==================================================
Total comments: 1,131,012
File size: 188.44 MB

Language Distribution:

Top 20 languages:
   1.     ru: 1,040,887 comments ( 92.03%)
   2. unknown:   26,240 comments (  2.32%)
   3.     uk:   18,996 comments (  1.68%)
   4.     en:   12,122 comments (  1.07%)
   5.     bg:    6,612 comments (  0.58%)
   6.     mk:    2,961 comments (  0.26%)
   7.     sr:    2,860 comments (  0.25%)
   8.     es:    2,365 comments (  0.21%)
   9.     de:    2,182 comments (  0.19%)
  10.     nl:    1,851 comments (  0.16%)
  11.     zh:    1,718 comments (  0.15%)
  12.     be:    1,082 comments (  0.10%)
  13.     ja:    1,035 comments (  0.09%)
  14.     fr:      816 comments (  0.07%)
  15.     mn:      680 comments (  0.06%)
  16.     tt:      623 comments (  0.06%)
  17.     ky:      578 comments (  0.05%)
  18.     pl:      559 comments (  0.05%)
  19.    sah:      526 comments (  0.05%)
  20.     it:      501 comments (  0.04%)

Summary:
  Total unique languages detected: 144
  Most common language: ru (1,040,887 comments, 92.03%)

Language Detection Confidence:
  Mean confidence score: 0.919
  Median confidence score: 0.995
  Comments with high confidence (>0.9): 952,519 (84.22%)
  Comments with low confidence (<0.5): 64,520 (5.70%)

==================================================
COMBINED STATISTICS
==================================================
Total comments across both platforms: 1,174,590

Top 10 languages across both platforms:
   1.     ru: 1,082,351.0 total (41,464.0 YouTube, 1,040,887.0 Telegram) -  92.15%
   2. unknown: 26,476.0 total ( 236.0 YouTube, 26,240.0 Telegram) -   2.25%
   3.     uk: 19,671.0 total ( 675.0 YouTube, 18,996.0 Telegram) -   1.67%
   4.     en: 12,388.0 total ( 266.0 YouTube, 12,122.0 Telegram) -   1.05%
   5.     bg:  6,728.0 total ( 116.0 YouTube, 6,612.0 Telegram) -   0.57%
   6.     mk:  3,023.0 total (  62.0 YouTube, 2,961.0 Telegram) -   0.26%
   7.     sr:  2,925.0 total (  65.0 YouTube, 2,860.0 Telegram) -   0.25%
   8.     es:  2,383.0 total (  18.0 YouTube, 2,365.0 Telegram) -   0.20%
   9.     de:  2,217.0 total (  35.0 YouTube, 2,182.0 Telegram) -   0.19%
  10.     nl:  1,862.0 total (  11.0 YouTube, 1,851.0 Telegram) -   0.16%
```


## Отбор комментариев
Берем только комментарии на русском, поскольку некторые методы неумеют работать с несколькими языками одновременно, поэтому используем только 1 язык. Самый большой - русский.
Файлы:
  - leave_only_russian.py

## Тематическое моделирование 

### BERTopic with ru-en-RoSBERTa backbone

В конце biblex на странице для цитирования https://maartengr.github.io/BERTopic/index.html#citation
Backbone ru-en-RoSBERTa
- Статья - https://arxiv.org/abs/2408.12503
- Модель на hugginface - https://huggingface.co/ai-forever/ru-en-RoSBERTa

# Оценка тематического моделирования 

TopMost https://topmost.readthedocs.io/en/latest/autoapi/index.html
Статья по ней https://arxiv.org/abs/2309.06908

Evaluating Dynamic Topic Models (https://arxiv.org/pdf/2309.08627)
Код к статье https://github.com/CharuJames/Evaluating-Dynamic-Topic-Models/tree/main



Понять как работают метрики в статье: https://arxiv.org/pdf/2309.08627
Взять D-LDA, D-ETM. DTM, DETM, CFDTM (https://github.com/bobxwu/TopMost/tree/main?tab=readme-ov-file#train-a-model), bertopic 
Получить главное топ 20 топиков, получить сравнения метрик для каждой модели в отдельности. Попробовать сравнить в руную результаты и метрики разных моделей, понять они чем-то схожи или нет. Как их можно будет сравнить между собой?


# 
запустили bertopic на базе "ai-forever/ru-en-RoSBERT" на 2 датасетах получили топики overtime. Они сохранились в папке bert_overtime.
-  dynamic_bertopic_with_metrics.py