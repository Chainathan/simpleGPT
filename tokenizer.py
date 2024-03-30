import tqdm

class CharacterTokenizer():
    def __init__(self, data):
        self.vocab = sorted(list(set(data)))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
    
    def encode(self, inp):
        return [self.char_to_idx[char] for char in inp]
    
    def decode(self, inp):
        return ''.join([self.idx_to_char[idx] for idx in inp])
    

class BPETokenizer():
    def __init__(self):
        self.vocab = None

    def _init_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.vocab_size = len(self.vocab)

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, num_merges, ids, verbose=False):  
        self.merges = {}
        for i in tqdm.tqdm(range(num_merges)):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose:
                tqdm.tqdm.write(f'{pair[0]}+{pair[1]}->{idx}')
        self._init_vocab()


    def decode(self, ids):
        if self.vocab is None:
            print("Vocab not initialized!")
            return None
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text 

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def save(self, file_prefix):
        model_file = file_prefix + ".bpe"
        with open(model_file, 'w') as f:
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    def load(self, model_file):
        assert model_file.endswith(".bpe")
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self._init_vocab()