import fasttext
import re
from nltk.tokenize import word_tokenize
"""
make all filter acts like transformation of torchvision
"""

__all__ = [
    "Compose",
    "Masker",
    "EmailMasker",
    "PhoneNumberMasker",
    "IPMasker",
    "GopherQualityFilter",
    "FastTextFilter"
]


class Compose:
    
    def __init__(self, filters: list[object], monitor: bool | None = None):
        self.filters = filters
        if monitor is not None:
            for f in filters:
                f.monitor = monitor
        
    def __call__(self, text: str) -> str:
        for f in self.filters:
            text = f(text)
            if text is None:
                return None
            if isinstance(text, (tuple, list)):
                text = text[0]
        return text
    
    def reset_monitor_info(self):
        for f in self.filters:
            f.reset_monitor_info()
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for f in self.filters:
            format_string += "\n"
            format_string += f"    {f}"
        format_string += "\n)"
        return format_string
    

class Masker:
    
    def __init__(self, replacement: str, pattern: str, monitor: bool = False):
        self.replacement = replacement
        self.pat = pattern
        self.monitor = monitor
        self.replace_count = 0
        
    def __call__(self, text):
        masked_text, count = re.subn(self.pat, self.replacement, text)
        if self.monitor:
            self.replace_count += count
        return masked_text, count
    
    def get_monitor_info(self):
        if self.monitor:
            return {"replace_count": self.replace_count}
        else:
            print("Monitor is not enabled.")
            return None
    
    def reset_monitor_info(self):
        if self.monitor:
            self.replace_count = 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(replacement={self.replacement}, pattern={self.pat})"
    

class EmailMasker(Masker):
    
    def __init__(self):
        super().__init__("|||EMAIL_ADDRESS|||",
                         r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    

class PhoneNumberMasker(Masker):
    
    def __init__(self):
        pattern = re.compile(
            r'''
            (?:(?:\+?1[\s\-.]*)?)         # 可选国家码
            (?:\(?\d{3}\)?)[\s\-\.]*      # 区号，可以有括号，后可跟空格、横线、点
            \d{3}[\s\-\.]*\d{4}           # 主体7位数字（3位+4位），中间分隔符可有
            ''', re.VERBOSE
        )
        super().__init__("|||PHONE_NUMBER|||", pattern)
        

class IPMasker(Masker):
    
    def __init__(self):
        octet = r'(?:25[0-5]|2[0-4][0-9]|1?\d\d?)'
        pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'
        super().__init__("|||IP_ADDRESS|||", pattern)
        

class GopherQualityFilter:
    
    def __init__(self,
                 min_num_word: int = 50,
                 max_num_word: int = 100000,
                 min_mean_num_char: int = 3,
                 max_mean_num_char: int = 10,
                 end_ellipsis_ratio: float = 0.3,
                 with_alpha_ratio: float = 0.8,
                 monitor: bool = False):
        self.min_num_word = min_num_word
        self.max_num_word = max_num_word
        self.min_mean_num_char = min_mean_num_char
        self.max_mean_num_char = max_mean_num_char
        assert end_ellipsis_ratio >= 0 and end_ellipsis_ratio <= 1
        self.end_ellipsis_ratio = end_ellipsis_ratio
        assert with_alpha_ratio >= 0 and with_alpha_ratio <= 1
        self.with_alpha_ratio = with_alpha_ratio
        self.monitor = monitor
        self.discard_cnt_by_word_num = 0
        self.discard_cnt_by_mean_word_len = 0
        self.discard_cnt_by_end_ellipsis = 0
        self.discard_cnt_by_without_alpha = 0
        self.discard_cnt = 0
        
    def __call__(self, text: str) -> str:
        words = word_tokenize(text)
        n = len(words)
        # Contain less than 50 or more than 100,000 words.
        if n < self.min_num_word or n > self.max_num_word:
            if self.monitor:
                self.discard_cnt_by_word_num += 1
                self.discard_cnt += 1
            return None
        
        # Have a mean word length outside the range of 3 to 10 characters.
        num_word = sum(len(word) for word in words)
        mean_word = num_word / n
        if mean_word < self.min_mean_num_char or mean_word > self.max_mean_num_char:
            if self.monitor:
                self.discard_cnt_by_mean_word_len += 1
                self.discard_cnt += 1
            return None
        
        # Have more than 30% of lines ending with an ellipsis (“...”).
        lines = text.splitlines()
        end_ellipsis_threshold = len(lines) * self.end_ellipsis_ratio
        end_ellipsis = 0
        for line in lines:
            if line.endswith("..."):
                end_ellipsis += 1
                if end_ellipsis > end_ellipsis_threshold:
                    if self.monitor:
                        self.discard_cnt_by_end_ellipsis += 1
                        self.discard_cnt += 1
                    return None
        
        # Contain less than 80% of words with at least one alphabetic character.
        without_alpha_threshold = n * (1 - self.with_alpha_ratio)
        without_alpha = 0
        for word in words:
            if not re.search('[a-zA-Z]', word):
                without_alpha += 1
                if without_alpha > without_alpha_threshold:
                    if self.monitor:
                        self.discard_cnt_by_without_alpha += 1
                        self.discard_cnt += 1
                    return None
        return text
    
    def get_monitor_info(self):
        if self.monitor:
            return {"discard_cnt_by_word_num": self.discard_cnt_by_word_num,
                    "discard_cnt_by_mean_word_len": self.discard_cnt_by_mean_word_len,
                    "discard_cnt_by_end_ellipsis": self.discard_cnt_by_end_ellipsis,
                    "discard_cnt_by_without_alpha": self.discard_cnt_by_without_alpha,
                    "discard_cnt": self.discard_cnt}
        else:
            print("Monitor is not enabled.")
            return None
    
    def reset_monitor_info(self):
        if self.monitor:
            self.discard_cnt_by_word_num = 0
            self.discard_cnt_by_mean_word_len = 0
            self.discard_cnt_by_end_ellipsis = 0
            self.discard_cnt_by_without_alpha = 0
            self.discard_cnt = 0
        
    def __repr__(self):
        return f"{self.__class__.__name__}(min_num_word={self.min_num_word}, max_num_word={self.max_num_word}, min_mean_num_char={self.min_mean_num_char}, max_mean_num_char={self.max_mean_num_char}, end_ellipsis_ratio={self.end_ellipsis_ratio}, with_alpha_ratio={self.with_alpha_ratio})"
    

class FastTextFilter:
    
    def __init__(self, model_path: str, target_label: str, threshold: float = 0.5, monitor: bool = False):
        self.model_path = model_path
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold
        self.target_label = target_label
        self.monitor = monitor
        self.discard_cnt = 0
        
    def get_monitor_info(self):
        if self.monitor:
            return {"discard_cnt": self.discard_cnt}
        else:
            print("Monitor is not enabled.")
            return None
    
    def reset_monitor_info(self):
        if self.monitor:
            self.discard_cnt = 0
        
    def predict(self, text: str, **kwds) -> tuple[str, float]:
        kwds['k'] = 1
        labels, probs = self.model.predict(text, **kwds)
        label = labels[0].replace("__label__", "")
        conf = float(probs[0])
        return label, conf
        
    def __call__(self, text: str, **kwds) -> str:
        label, conf = self.predict(text, **kwds)
        if label != self.target_label:
            if self.monitor:
                self.discard_cnt += 1
            return None
        if conf < self.threshold:
            if self.monitor:
                self.discard_cnt += 1
            return None
        return text
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_path={self.model_path}, target_label={self.target_label}, threshold={self.threshold})"