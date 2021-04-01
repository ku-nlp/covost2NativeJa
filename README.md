# CoVoST 2 Native Japanese Dataset

## Overview
This corpus is a corpus for speech-to-text translation (ST) in Japanese-English.
It is based on [CoVoST 2 dataset](https://github.com/facebookresearch/covost#covost-2), a large-scale multilingual ST corpus including translations from 21 languages to English and English to 15 languages.
We checked the audio of the Japanese dataset in CoVoST 2 and found that about 30% of the utterances were apparently spoken by non-native speakers.
Although the presence of various accents itself is a useful feature for a speech corpus, it is possible for too many utterances by non-native speakers to affect ST performance.
Therefore, we rerecorded the speech of all 2,438 utterances from the CoVoST 2 Japanese dataset by 7 male Japanese native speakers.
Text of this dataset is identical to the original CoVoST 2 dataset.
The distribution of the training, development, and test sets is also identical to the CoVoST 2 dataset:

| Train | Development | Test |
| --- | --- | --- |
| 1,119 | 635 | 684 |

## Model training
You can use the [fairseq S2T toolkit](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text) for pre-processing, training, and evaluation.

- Data download
```bash
mkdir ${COVOST_ROOT}/ja
cd ${COVOST_ROOT}
git clone https://github.com/ku-nlp/covost2NativeJa.git
tar zxvf clips.tar.gz
```

- Prepare fairseq
	- Follow the instructions of [fairseq installation](https://github.com/pytorch/fairseq#requirements-and-installation).

- Preprocessing
```bash
python ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_covost_data.py \
	--data-root ${COVOST_ROOT} --vocab-type char \
	--src-lang ja --tgt-lang en
```

- Training
```bash
fairseq-train ${COVOST_ROOT}/ja \
  --config-yaml config_st_ja_en.yaml --train-subset train_st_ja_en --valid-subset dev_st_ja_en \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-update 60000 --max-tokens 50000
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --encoder-freezing-updates 1000 --optimizer adam --lr 2e-3 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 # update-freq shoule be changed according to the number of GPUs you use
```

- Evaluation
```bash
fairseq-generate ${COVOST_ROOT}/ja \
	--config-yaml config_st_ja_en.yaml --gen-subset test_st_ja_en --task speech_to_text \
	--path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
	--max-tokens 50000 --beam 5 --scoring sacrebleu
```

## License
The license for this corpus is subject to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
