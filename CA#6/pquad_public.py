import json
import datasets
_CITATION = """\
@article{darvishi2022pquad,
  title={PQuAD: A Persian Question Answering Dataset},
  author={Darvishi, Kasra and Shahbodagh, Newsha and Abbasiantaeb, Zahra and Momtazi, Saeedeh},
  journal={arXiv preprint arXiv:2202.06219},
  year={2022}
}
"""
_DESCRIPTION = """\\\\
ParSQuAD: Persian Question Answering Dataset based on Machine Translation of SQuAD 2.0
"""
_URL = "https://raw.githubusercontent.com/vassef/pquad_public/main/"
_URLS = {
    "train": _URL + "train_samples.json",
    "validation":_URL + "validation_samples.json",
    "test": _URL + "test_samples.json",
}
class pquad_public_Config(datasets.BuilderConfig):
    """BuilderConfig for PQuAD."""
    def __init__(self, **kwargs):
        """BuilderConfig for PQuAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(pquad_public_Config, self).__init__(**kwargs)
class pquad_public(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        pquad_public_Config(name="pquad_public", version=datasets.Version("1.0.0"), description="PQuAD plaint text version 2"),
    ]
    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("float64"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/vassef/pquad_public/",
            citation=_CITATION,
        )
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(persian_qa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]})
            ]
    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(persian_qa): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            print(filepath)
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "").strip()
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
