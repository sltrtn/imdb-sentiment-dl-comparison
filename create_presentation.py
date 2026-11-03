"""Create the project presentation using the supplied ITR deck as the visual template."""

from pathlib import Path
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parent
TEMPLATE = Path("/home/mad/Downloads/ITR/Kriaa Performance Capture Manager — Industrial Training.pptx")
OUTPUT = Path("/home/mad/Downloads/DLProject/IMDB_Sentiment_DL_Presentation.pptx")

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def put_text(root, shape_id, value):
    shape = root.xpath(f".//p:sp[p:nvSpPr/p:cNvPr/@id='{shape_id}']", namespaces=NS)[0]
    nodes = shape.xpath(".//a:t", namespaces=NS)
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


def put_table(root, rows):
    table = root.xpath(".//a:tbl", namespaces=NS)[0]
    for tr, values in zip(table.xpath("./a:tr", namespaces=NS), rows):
        cells = tr.xpath("./a:tc", namespaces=NS)
        for cell, value in zip(cells, values):
            text_nodes = cell.xpath(".//a:t", namespaces=NS)
            text_nodes[0].text = value
            for node in text_nodes[1:]:
                node.text = ""


SLIDES = {
    1: {
        2: "DEEP LEARNING PROJECT · ICT-4442",
        3: "IMDB Sentiment\nDeep Learning Comparison",
        4: "A controlled comparison of MLP, TextCNN, BiLSTM and fine-tuned DistilBERT for movie-review sentiment classification.",
        5: "Hasil Haroon · T Rama Mohan Aniketh · Madhavan K · Ayrton Pulikottil",
        6: "230953019 · 230911224 · 230953298 · 230953448",
        7: "Manipal Institute of Technology",
        8: "School of Computer Engineering · November 2026",
    },
    2: {
        2: "PROJECT AT A GLANCE",
        3: "One Dataset. Four Model Families.",
        5: "DATASET",
        6: "IMDB Large Movie Review Dataset · 50,000 labelled reviews",
        8: "TASK",
        9: "Binary sentiment classification: positive vs negative",
        11: "COMMON PROTOCOL",
        12: "22,500 train · 2,500 validation · 25,000 held-out test",
        14: "QUESTION",
        15: "How much accuracy is worth the extra model complexity and training cost?",
        16: "IMDB Sentiment DL Comparison · ICT-4442",
        17: "02 / 12",
    },
    3: {
        2: "WHY THIS PROBLEM",
        3: "Two Ideas Behind the Comparison",
        5: "Sentiment is contextual",
        6: "Review language is informal and mixed. Word order changes meaning: “not good” and “not bad” contain similar tokens but express opposite sentiment.",
        8: "Fair comparison matters",
        9: "Every model sees the same cleaned data, fixed split and metrics. Validation F1 selects a configuration; the official test set is used only once.",
        10: "IMDB Sentiment DL Comparison · ICT-4442",
        11: "03 / 12",
    },
    4: {
        2: "OBJECTIVES & SCOPE",
        3: "A Reproducible, Leakage-Free Study",
        4: "We complete the scope proposed in the synopsis and interim report: implement four architectures, tune each on validation data, compare their test performance and cost, and inspect failures of the strongest model.",
        6: "DATA SOURCE",
        7: "Official IMDB mirror",
        8: "CLEANING",
        9: "HTML removal + whitespace normalization",
        10: "MODEL SELECTION",
        11: "Validation F1 only",
        12: "REPRODUCIBILITY",
        13: "Seed 42 · JSON results · GitHub repository",
        14: "IMDB Sentiment DL Comparison · ICT-4442",
        15: "04 / 12",
    },
    5: {
        2: "EVALUATION DESIGN",
        3: "What Makes the Comparison Trustworthy",
        4: "01",
        5: "The official 25,000-review test partition is never used for tuning.",
        6: "02",
        7: "TF--IDF vocabularies and neural token vocabularies are fitted on training text only.",
        8: "03",
        9: "Each family uses a small validation-based configuration search, not a test-set search.",
        10: "04",
        11: "Accuracy, precision, recall, F1, ROC--AUC and confusion matrices are reported together.",
        12: "05",
        13: "Saved result JSON files, plots and scripts make the experiment reproducible.",
        14: "IMDB Sentiment DL Comparison · ICT-4442",
        15: "05 / 12",
    },
    6: {
        2: "METHOD",
        3: "One Shared Experimental Workflow",
        4: "IMDB reviews → clean & split",
        7: "Train split",
        8: "22,500 reviews · vocabularies fitted here",
        10: "Validation split",
        11: "2,500 reviews · choose highest F1 configuration",
        13: "Test split",
        14: "25,000 untouched reviews · final evaluation only",
        16: "MLP · TextCNN · BiLSTM · DistilBERT → selected model scores are directly comparable",
        17: "IMDB Sentiment DL Comparison · ICT-4442",
        18: "06 / 12",
    },
    8: {
        2: "RESULTS",
        3: "DistilBERT Leads the Comparison",
        4: "Best overall result",
        5: "Fine-tuned DistilBERT: 0.9141 accuracy, 0.9138 F1 and 0.9723 ROC--AUC. It improves accuracy by 3.7 percentage points over the strongest non-transformer.",
        6: "Strong lightweight baseline",
        7: "The TF--IDF MLP reaches 0.8774 accuracy. It demonstrates that sparse lexical features remain highly competitive on IMDB.",
        8: "GloVe matters",
        9: "TextCNN improves from 0.8408 with random embeddings to 0.8718 with GloVe and 300-token inputs.",
        10: "Trade-off",
        11: "TextCNN trains in 24 s; DistilBERT takes 643 s on the selected run. Higher accuracy has a clear compute cost.",
        14: "IMDB Sentiment DL Comparison · ICT-4442",
        15: "08 / 12",
    },
    9: {
        2: "ERROR ANALYSIS",
        3: "Where Does the Best Model Struggle?",
        6: "Review length",
        7: "Accuracy is 0.9340 for the shortest quartile, but falls to 0.8647 for reviews of 274–2,235 words.",
        8: "Negation cues",
        9: "Reviews with “not”, “never”, “no” or similar cues: 0.9081 accuracy; without them: 0.9322.",
        10: "Why?",
        11: "Long, mixed reviews are harder and evidence late in a review can be lost after the 256-token limit.",
        12: "Qualitative cases",
        13: "Several high-confidence apparent errors contain language that conflicts with the assigned label, suggesting noisy or mixed reviews.",
        14: "IMDB Sentiment DL Comparison · ICT-4442",
        15: "09 / 12",
    },
    10: {
        2: "KEY TAKEAWAYS",
        3: "What the Experiment Shows",
        5: "Pretraining wins",
        6: "DistilBERT is the strongest across accuracy, F1 and ROC--AUC because it begins with contextual language knowledge.",
        8: "Simple baselines matter",
        9: "The MLP is only 3.7 percentage points behind DistilBERT while requiring far less tuning time and capacity.",
        11: "Embeddings help",
        12: "GloVe substantially lifts TextCNN and fine-tuned GloVe gives the best BiLSTM configuration.",
        13: "Result selection is validation-only; final figures, tables, scripts and results JSON are committed to GitHub.",
        14: "IMDB Sentiment DL Comparison · ICT-4442",
        15: "10 / 12",
    },
    11: {
        2: "CONCLUSION",
        3: "On one fixed IMDB protocol, fine-tuned DistilBERT delivers the best sentiment classification result — while the MLP and GloVe TextCNN remain strong, lower-cost choices.",
        4: "The project turns an architectural choice into an evidence-based trade-off: 0.9141 accuracy for DistilBERT versus 0.8774 for MLP, with reproducible code, plots and error analysis.",
    },
    12: {
        2: "Thank You",
        3: "Questions & discussion",
        4: "Hasil Haroon · T Rama Mohan Aniketh · Madhavan K · Ayrton Pulikottil",
        5: "GitHub: github.com/sltrtn/imdb-sentiment-dl-comparison · MIT Manipal",
    },
}

TABLE = [
    ["Model", "Representation", "Selected configuration", "Test performance"],
    ["MLP", "20k TF--IDF uni/bi-grams", "128 hidden · L2=1e-3", "Acc 0.8774 · F1 0.8773"],
    ["TextCNN", "Random embeddings", "200 tokens · 100 filters", "Acc 0.8408 · F1 0.8529"],
    ["TextCNN", "Fine-tuned GloVe", "300 tokens · 128 filters", "Acc 0.8718 · F1 0.8753"],
    ["BiLSTM", "Fine-tuned GloVe", "128 hidden · 256 tokens", "Acc 0.8668 · F1 0.8698"],
    ["DistilBERT", "Pretrained transformer", "256 tokens · lr 2e-5", "Acc 0.9141 · F1 0.9138"],
    ["Selection", "Validation F1", "Seed 42 · held-out test", "AUC best: 0.9723"],
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(TEMPLATE) as source, ZipFile(OUTPUT, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                number = int(item.filename.rsplit("slide", 1)[1].split(".xml", 1)[0])
                root = etree.fromstring(payload)
                if number in SLIDES:
                    for shape_id, text in SLIDES[number].items():
                        put_text(root, shape_id, text)
                if number == 7:
                    put_text(root, 2, "MODELS & TUNING")
                    put_text(root, 3, "Four Architectures, One Protocol")
                    put_text(root, 5, "IMDB Sentiment DL Comparison · ICT-4442")
                    put_text(root, 6, "07 / 12")
                    put_table(root, TABLE)
                payload = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            elif item.filename == "ppt/media/image-8-1.png":
                payload = (ROOT / "results/figures/metric_comparison.png").read_bytes()
            elif item.filename == "ppt/media/image-9-1.jpeg":
                image = Image.open(ROOT / "results/figures/accuracy_by_length.png").convert("RGB")
                out = BytesIO()
                image.save(out, "JPEG", quality=94)
                payload = out.getvalue()
            target.writestr(item, payload)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
