import ftfy
from omegaconf import OmegaConf
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    models,
    processors,
    decoders,
    trainers,
)


def fix_text(text):
    return ftfy.fix_text(
        text,
        unescape_html=True,
        remove_terminal_escapes=True,
        fix_encoding=True,
        replace_lossy_sequences=True,
        decode_inconsistent_utf8=True,
        fix_latin_ligatures=True,
        fix_character_width=True,
        uncurl_quotes=True,
        fix_line_breaks=True,
        remove_control_chars=True,
        normalization="NFKD",
        explain=False,
    )


def build_tokenizer(cfg):
    model = models.WordPiece(
        unk_token="[UNK]", max_input_chars_per_word=cfg.tokenizer.max_chars_per_word
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKD(),  # compatibility decomposed form to minimize number of distinct characters
            normalizers.StripAccents(),
            normalizers.Lowercase(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),  # and punctuation boundary, "hi!! !" -> ["hi", "!!", "!"]
            pre_tokenizers.Punctuation(behavior="isolated"),  # "???" -> ["?", "?", "?"]
            pre_tokenizers.Digits(individual_digits=True),  # "911" -> ["9", "1", "1"]
        ]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cfg.tokenizer.special_tokens.cls),
            ("[SEP]", cfg.tokenizer.special_tokens.sep),
        ],
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    return tokenizer


if __name__ == "__main__":
    from tqdm import tqdm
    from datasets import load_dataset

    cfg = OmegaConf.load("config.yaml")
    tokenizer = build_tokenizer(cfg)

    trainer = trainers.WordPieceTrainer(
        vocab_size=cfg.tokenizer.vocab_size,  # keep merging until this size is reached
        min_frequency=cfg.tokenizer.min_token_frequency,
        limit_alphabet=cfg.tokenizer.max_alphabet_size,
        special_tokens=["[CLS]", "[SEP]", "[MASK]", "[UNK]"],
        continuing_subword_prefix="##",
        show_progress=True,
    )

    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    fix_article = fix_text if cfg.tokenizer.fix_text else lambda t: t

    tokenizer.train_from_iterator(
        (
            line
            for article in tqdm(
                dataset, total=dataset.info.splits["train"].num_examples
            )
            for line in fix_article(article["text"]).split("\n")
        ),
        trainer=trainer,
    )
    tokenizer.save(cfg.tokenizer.checkpoint_path)
