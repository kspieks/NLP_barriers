import re
from typing import List

from transformers import BertTokenizer


SMI_REGEX_PATTERN =  r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

class SmilesTokenizer(BertTokenizer):
    """
    Constructs a SmilesBertTokenizer.
    Adapted from https://github.com/rxn4chemistry/rxnfp
    
    Args:
        vocab_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        do_lower_case=False,
        **kwargs,
    ) -> None:
        """
        Constructs an SmilesTokenizer.
        
        Args:
            vocab_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        
        # define tokenization utilities
        self.tokenizer = RegexTokenizer()

    @property
    def vocab_list(self) -> List[str]:
        """
        List vocabulary tokens.

        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text representing an enzymatic reaction with AA sequence information.
        
        Args:
            text: text to tokenize.
        
        Returns:
            extracted tokens.
        """
        return self.tokenizer.tokenize(text)


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str=SMI_REGEX_PATTERN) -> None:
        """
        Constructs a RegexTokenizer.
        
        Args:
            regex_pattern: regex pattern used for tokenization.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """
        Regex tokenization.
        
        Args:
            text: text to tokenize.
        
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens

