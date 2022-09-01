import json
import datasets
_CITATION = """\
@ARTICLE{Abadani_undated-pf,
  title   = "{ParSQuAD}: Persian Question Answering Dataset based on Machine
             Translation of {SQuAD} 2.0",
  author  = "Abadani, N and Mozafari, J and Fatemi, A and Nematbakhsh, M and
             Kazemi, A",
  journal = "International Journal of Web Research",
  volume  =  4,
  number  =  1
}
"""
mode = input('\nPlease Enter your desire mode(manual / automatic) :')
while mode not in ['manual','automatic']:
    #if mode not in ['manual','automatic']:
    print('\nInvalid mode')
    mode = input('Enter Again :')
_DESCRIPTION = """\\\\
ParSQuAD: Persian Question Answering Dataset based on Machine Translation of SQuAD 2.0
"""
_URL = "https://raw.githubusercontent.com/vassef/ParSQuad/main/"
_URLS = {
    "manual-train": _URL + "ParSQuAD-manual-train.json",
    "automatic-train":_URL + "ParSQuAD-automatic-train.json",
    "manual-dev": _URL + "ParSQuAD-manual-dev.json",
    "automatic-dev":_URL + "ParSQuAD-automatic-dev.json",
}
class ParSQuADConfig(datasets.BuilderConfig):
    """BuilderConfig for PersianQA."""
    def __init__(self, **kwargs):
        """BuilderConfig for PersianQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ParSQuADConfig, self).__init__(**kwargs)
class ParSQuAD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ParSQuADConfig(name="ParSQuAD", version=datasets.Version("1.0.0"), description="ParSQuAD plaint text version 2"),
    ]
    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
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
            homepage="https://github.com/vassef/ParSQuad/",
            citation=_CITATION,
        )
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(persian_qa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        if mode == 'manual':
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["manual-train"]}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["manual-dev"]})
                ]
        else:
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["automatic-train"]}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["automatic-dev"]})
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
