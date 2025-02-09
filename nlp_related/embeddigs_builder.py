"""
Optimized script for building sketch‐based document embeddings and creating indexes.

This module implements Count–Min Sketch (CMS), CountSketch (CS), and a
SIDF aggregator to approximate document frequencies. The IndexEmbeddingBuilder
class builds document embeddings based on either TF–IDF or BM25 weighting,
distributing text files into bins and processing them in parallel.
It also provides static methods for loading embeddings, building Faiss indexes
(L2 and cosine), running Isolation Forest outlier detection, and computing the
correlation between distances from the CMS and CS embeddings.
"""

import os
import re
import math
import random
import json
import multiprocessing
import numpy as np
import mmh3
import srsly
import faiss  # pip install faiss-cpu (or faiss-gpu)
from itertools import chain
from typing import List, Tuple, Dict, Any, Literal
from sklearn.ensemble import IsolationForest  # pip install scikit-learn

# Precompile token‐splitting regex
TOKEN_PATTERN = re.compile(r"\W+")

###############################################################################
# Module‐level globals for multiprocessing workers (for sketch and embedding params)
###############################################################################
# Globals for sketch building
CMS_DIM: Tuple[int, int] = None  # (width, depth): width = number of columns, depth = number of rows
CS_DIM: Tuple[int, int] = None
SEEDS: Dict[str, List[int]] = None  # expects keys "cms", "cs", "cs_sign"

# Globals for embedding building
SIDF_CMS = None
SIDF_CS = None
NUM_DOCS: int = None
AVG_DL: float = None
EMB_SEED: int = None
DIMENSION: int = None
MODEL: Literal["bm25", "tfidf"] = None
K: float = None
B: float = None


###############################################################################
# Sketch Classes: CMS, CS, & SIDF
###############################################################################

class BaseSketch:
    """
    Common interface for Count–Min Sketch and CountSketch.
    """

    def update(self, element: str, count: int = 1) -> None:
        """
        Update the sketch count for the given element.

        Parameters
        ----------
        element : str
            The element to update.
        count : int, optional
            The increment (default is 1).
        """
        raise NotImplementedError

    def query(self, element: str) -> float:
        """
        Query the sketch for the count estimate of the given element.

        Parameters
        ----------
        element : str
            The element to query.

        Returns
        -------
        float
            The estimated count.
        """
        raise NotImplementedError

    def get_rows(self) -> int:
        """
        Return the number of rows in the sketch.

        Returns
        -------
        int
            The number of rows.
        """
        raise NotImplementedError

    def get_cols(self) -> int:
        """
        Return the number of columns in the sketch.

        Returns
        -------
        int
            The number of columns.
        """
        raise NotImplementedError

    def get_seeds(self) -> List[int]:
        """
        Return the list of hash seeds used.

        Returns
        -------
        List[int]
            The list of seeds.
        """
        raise NotImplementedError

    def get_count_matrix(self) -> np.ndarray:
        """
        Return the internal count matrix.

        Returns
        -------
        np.ndarray
            The count matrix.
        """
        raise NotImplementedError


class CMS(BaseSketch):
    """
    Count–Min Sketch (CMS).

    The sketch is parameterized by:
      - rows: depth (number of hash functions)
      - cols: width (range of each hash function)
      - seeds: one seed per row
    """

    def __init__(self, rows: int, cols: int, seeds: List[int] = None) -> None:
        """
        Initialize a Count–Min Sketch.

        Parameters
        ----------
        rows : int
            Number of rows (depth).
        cols : int
            Number of columns (width).
        seeds : List[int], optional
            Optional list of distinct hash seeds.

        Raises
        ------
        ValueError
            If not enough seeds are provided or seeds are not distinct.
        """
        self.rows = rows
        self.cols = cols
        if seeds is None:
            self.seeds = [i + 1 for i in range(rows)]
        else:
            if len(seeds) < rows:
                raise ValueError("Not enough seeds for the given number of rows.")
            if len(set(seeds)) != len(seeds):
                raise ValueError("All seeds must be distinct.")
            self.seeds = seeds
        self.count_matrix = np.zeros((rows, cols), dtype=np.uint32)

    def update(self, element: str, count: int = 1) -> None:
        """
        Increment the count for `element` by `count`.

        Parameters
        ----------
        element : str
            The element to update.
        count : int, optional
            The count to add (default is 1).
        """
        for r in range(self.rows):
            c = mmh3.hash(element, self.seeds[r]) % self.cols
            self.count_matrix[r, c] += count

    def query(self, element: str) -> float:
        """
        Estimate the count for `element` by taking the minimum value across rows.

        Parameters
        ----------
        element : str
            The element to query.

        Returns
        -------
        float
            The estimated count.
        """
        min_val = float("inf")
        for r in range(self.rows):
            c = mmh3.hash(element, self.seeds[r]) % self.cols
            min_val = min(min_val, self.count_matrix[r, c])
        return float(min_val)

    def get_rows(self) -> int:
        """Return the number of rows in the sketch."""
        return self.rows

    def get_cols(self) -> int:
        """Return the number of columns in the sketch."""
        return self.cols

    def get_seeds(self) -> List[int]:
        """Return the list of hash seeds used."""
        return self.seeds

    def get_count_matrix(self) -> np.ndarray:
        """Return the internal count matrix."""
        return self.count_matrix

    @staticmethod
    def compute_width_depth(epsilon: float, delta: float) -> Tuple[int, int]:
        """
        Compute the CMS width and depth parameters given error and probability bounds.

        Typically:
          - width w ~ O(1/epsilon)
          - depth d ~ O(log(1/delta))

        For demonstration, we use:
          w = ceil(e / epsilon^2), d = ceil(log(1/delta))

        Parameters
        ----------
        epsilon : float
            The error parameter.
        delta : float
            The probability parameter.

        Returns
        -------
        Tuple[int, int]
            A tuple containing (width, depth).
        """
        w = math.ceil(math.e / (epsilon ** 2))
        d = math.ceil(math.log(1 / delta))
        return w, d  # (width, depth)


class CS(BaseSketch):
    """
    CountSketch (CS) with sign-hashing.

    The sketch is parameterized by:
      - rows: depth
      - cols: width
      - seeds: one seed per row for index hashing
      - sign_seeds: one seed per row for sign determination
    """

    def __init__(self, rows: int, cols: int, seeds: List[int] = None, sign_seeds: List[int] = None) -> None:
        """
        Initialize a CountSketch.

        Parameters
        ----------
        rows : int
            Number of rows (depth).
        cols : int
            Number of columns (width).
        seeds : List[int], optional
            Optional list of distinct hash seeds.
        sign_seeds : List[int], optional
            Optional list of distinct sign hash seeds.

        Raises
        ------
        ValueError
            If not enough seeds or sign_seeds are provided or they are not distinct.
        """
        self.rows = rows
        self.cols = cols
        if seeds is None:
            self.seeds = [i + 1 for i in range(rows)]
        else:
            if len(seeds) < rows:
                raise ValueError("Not enough seeds.")
            if len(set(seeds)) != len(seeds):
                raise ValueError("All seeds must be distinct.")
            self.seeds = seeds

        if sign_seeds is None:
            self.sign_seeds = [rows + i + 1 for i in range(rows)]
        else:
            if len(sign_seeds) < rows:
                raise ValueError("Not enough sign_seeds.")
            if len(set(sign_seeds)) != len(sign_seeds):
                raise ValueError("All sign seeds must be distinct.")
            self.sign_seeds = sign_seeds

        self.count_matrix = np.zeros((rows, cols), dtype=np.int64)

    def _sign(self, element: str, row: int) -> int:
        """
        Compute the sign for `element` at the given row.

        Parameters
        ----------
        element : str
            The element for which to compute the sign.
        row : int
            The row index.

        Returns
        -------
        int
            1 if the hash is even, -1 otherwise.
        """
        h = mmh3.hash(element, self.sign_seeds[row])
        return 1 if (h & 1) == 0 else -1

    def update(self, element: str, count: int = 1) -> None:
        """
        Update the sketch count for `element` with the appropriate sign.

        Parameters
        ----------
        element : str
            The element to update.
        count : int, optional
            The count to add (default is 1).
        """
        for r in range(self.rows):
            c = mmh3.hash(element, self.seeds[r]) % self.cols
            s = self._sign(element, r)
            self.count_matrix[r, c] += s * count

    def query(self, element: str) -> float:
        """
        Estimate the count for `element` using the median of signed counts.

        Parameters
        ----------
        element : str
            The element to query.

        Returns
        -------
        float
            The estimated count.
        """
        estimates = []
        for r in range(self.rows):
            c = mmh3.hash(element, self.seeds[r]) % self.cols
            s = self._sign(element, r)
            estimates.append(s * self.count_matrix[r, c])
        return float(np.median(estimates))

    def get_rows(self) -> int:
        """Return the number of rows in the sketch."""
        return self.rows

    def get_cols(self) -> int:
        """Return the number of columns in the sketch."""
        return self.cols

    def get_seeds(self) -> List[int]:
        """Return the list of hash seeds used."""
        return self.seeds

    def get_sign_seeds(self) -> List[int]:
        """Return the list of sign hash seeds used."""
        return self.sign_seeds

    def get_count_matrix(self) -> np.ndarray:
        """Return the internal count matrix."""
        return self.count_matrix

    @staticmethod
    def compute_width_depth(epsilon: float, delta: float) -> Tuple[int, int]:
        """
        Compute the CS width and depth parameters given error and probability bounds.

        For CountSketch:
          - width w ~ O(1/epsilon^2)
          - depth d ~ O(log(1/delta))

        Parameters
        ----------
        epsilon : float
            The error parameter.
        delta : float
            The probability parameter.

        Returns
        -------
        Tuple[int, int]
            A tuple containing (width, depth).
        """
        w = math.ceil(1.0 / (epsilon ** 2))
        d = math.ceil(math.log(1.0 / delta))
        return w, d  # (width, depth)


def _array_to_bytes(arr: np.ndarray) -> Dict[str, Any]:
    """
    Serialize a NumPy array to a dictionary with base64-encoded data.

    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to serialize.

    Returns
    -------
    dict
        A dictionary containing "shape", "dtype", and "data".
    """
    import base64
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("ascii")
    }


def _array_from_bytes(d: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize a NumPy array from the dictionary produced by `_array_to_bytes`.

    Parameters
    ----------
    d : dict
        A dictionary with keys "shape", "dtype", and "data".

    Returns
    -------
    np.ndarray
        The reconstructed NumPy array.
    """
    import base64
    shape = tuple(d["shape"])
    dtype = np.dtype(d["dtype"])
    raw = base64.b64decode(d["data"])
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class SIDF:
    """
    Sketch‐based Inverse Document Frequency (IDF) aggregator.

    This class aggregates counts from multiple sketches (all CMS or all CS)
    to estimate document frequency. It also provides TF–IDF or BM25 style measures.
    """

    def __init__(self, doc_count: int, sketches: List[BaseSketch]) -> None:
        """
        Initialize SIDF with the total document count and a list of sketches.

        Parameters
        ----------
        doc_count : int
            Total number of documents.
        sketches : List[BaseSketch]
            List of CMS or CS sketch instances.

        Raises
        ------
        ValueError
            If no sketches are provided or if sketches are mixed types.
        """
        self.doc_count = doc_count
        if not sketches:
            raise ValueError("No sketches provided to SIDF.")

        first = sketches[0]
        self.rows = first.get_rows()
        self.cols = first.get_cols()
        self.seeds = first.get_seeds()

        if isinstance(first, CMS):
            self.mode = "CMS"
            # Ensure all sketches are CMS
            for sk in sketches:
                if not isinstance(sk, CMS):
                    raise ValueError("All sketches must be CMS or all must be CS.")
            mats = np.stack([sk.get_count_matrix() for sk in sketches], axis=2)
            self.count_matrix = np.sum(mats, axis=2).astype(np.uint32)
            self.sign_seeds = None
        elif isinstance(first, CS):
            self.mode = "CS"
            self.sign_seeds = first.get_sign_seeds()
            for sk in sketches:
                if not isinstance(sk, CS):
                    raise ValueError("All sketches must be CS or all must be CMS.")
                if sk.get_sign_seeds() != self.sign_seeds:
                    raise ValueError("CS sign_seeds mismatch.")
            mats = np.stack([sk.get_count_matrix() for sk in sketches], axis=2)
            self.count_matrix = np.sum(mats, axis=2).astype(np.int64)
        else:
            raise ValueError("Sketches must be CMS or CS.")

    def doc_count_value(self) -> int:
        """
        Return the total number of documents.

        Returns
        -------
        int
            The document count.
        """
        return self.doc_count

    def query(self, term: str) -> float:
        """
        Estimate the number of documents that contain `term` using the underlying sketch.

        Parameters
        ----------
        term : str
            The term to query.

        Returns
        -------
        float
            The estimated document frequency.
        """
        if self.mode == "CMS":
            return self._cms_query(term)
        else:
            return self._cs_query(term)

    def _cms_query(self, term: str) -> float:
        """
        Query using the CMS method.

        Parameters
        ----------
        term : str
            The term to query.

        Returns
        -------
        float
            The minimum count across rows.
        """
        min_val = float("inf")
        for r in range(self.rows):
            c = mmh3.hash(term, self.seeds[r]) % self.cols
            min_val = min(min_val, self.count_matrix[r, c])
        return float(min_val)

    def _cs_query(self, term: str) -> float:
        """
        Query using the CS method.

        Parameters
        ----------
        term : str
            The term to query.

        Returns
        -------
        float
            The median of signed counts.
        """
        estimates = []
        for r in range(self.rows):
            c = mmh3.hash(term, self.seeds[r]) % self.cols
            sign_val = mmh3.hash(term, self.sign_seeds[r])
            s = 1 if (sign_val & 1) == 0 else -1
            estimates.append(s * self.count_matrix[r, c])
        return float(np.median(estimates))

    def __getitem__(self, term: str) -> float:
        """
        Compute a weight for the term based on the document frequency.

        For "tfidf": log(doc_count / presence)
        For "bm25": log((doc_count - presence + 0.5) / (presence + 0.5 + 1e-9))
        Returns 0 if the term is not present in any document.

        Parameters
        ----------
        term : str
            The term for which to compute the weight.

        Returns
        -------
        float
            The computed weight.
        """
        presence = self.query(term)
        if presence < 1:
            return 0.0
        if MODEL == "bm25":
            # For BM25, cap presence at doc_count to avoid non-positive numerator.
            presence = min(presence, self.doc_count)
            return math.log((self.doc_count - presence + 0.5) / (presence + 0.5 + 1e-9))
        elif MODEL == "tfidf":
            return math.log(self.doc_count / presence)
        return math.log(self.doc_count / presence)

    def save(self, out_dir: str, filename: str) -> None:
        """
        Serialize and save the SIDF object to a msgpack file.

        Parameters
        ----------
        out_dir : str
            Directory to save the file.
        filename : str
            Filename for the saved SIDF.
        """
        path = os.path.join(out_dir, filename)
        data = {
            "doc_count": self.doc_count,
            "rows": self.rows,
            "cols": self.cols,
            "mode": self.mode,
            "seeds": self.seeds,
            "count_matrix": _array_to_bytes(self.count_matrix)
        }
        if self.mode == "CS" and self.sign_seeds is not None:
            data["sign_seeds"] = self.sign_seeds
        srsly.write_msgpack(path, data)

    @staticmethod
    def load(filepath: str) -> "SIDF":
        """
        Load a SIDF object from a msgpack file.

        Parameters
        ----------
        filepath : str
            Path to the msgpack file.

        Returns
        -------
        SIDF
            The reconstructed SIDF object.
        """
        data = srsly.read_msgpack(filepath)
        sidf = SIDF.__new__(SIDF)
        sidf._from_data(data)
        return sidf

    def _from_data(self, data: Dict[str, Any]) -> None:
        """
        Initialize the SIDF object from a data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing SIDF parameters.
        """
        self.doc_count = data["doc_count"]
        self.rows = data["rows"]
        self.cols = data["cols"]
        self.mode = data["mode"]
        self.seeds = data["seeds"]
        self.count_matrix = _array_from_bytes(data["count_matrix"])
        self.sign_seeds = data.get("sign_seeds", None)


ModelCateg = Literal["bm25", "tfidf"]

###############################################################################
# IndexEmbeddingBuilder Class
###############################################################################

class IndexEmbeddingBuilder:
    """
    Build document embeddings using sketch‐based approximations of IDF and
    create Faiss indexes to enable nearest‐neighbor queries.

    This class distributes text files into bins, builds local sketches (CMS and CS)
    in parallel, merges them into global SIDF objects, and then builds document embeddings
    using either TF–IDF or BM25 weighting. It also provides static methods for loading
    embeddings, building Faiss indexes (L2 and cosine), running Isolation Forest for
    outlier detection, and computing the correlation between distances from the CMS and CS embeddings.
    """

    def __init__(self,
                 src_path: str,
                 tmp_path: str,
                 out_json: str,
                 out_npz: str,
                 epsilon: float = 0.01,
                 delta: float = 0.01,
                 num_procs: int = 9,
                 bin_size: int = 1000,
                 model: ModelCateg = "bm25",
                 k: float = 1.2,
                 b: float = 0.75,
                 emb_seed: int = 42,
                 dimension: int = 300) -> None:
        """
        Initialize the IndexEmbeddingBuilder.

        Parameters
        ----------
        src_path : str
            Path to the source directory containing .txt files.
        tmp_path : str
            Temporary directory path for intermediate files.
        out_json : str
            Filename to save document filenames.
        out_npz : str
            Filename to save the final embeddings (.npz file).
        epsilon : float, optional
            Error parameter for sketch construction (default is 0.01).
        delta : float, optional
            Probability parameter for sketch construction (default is 0.01).
        num_procs : int, optional
            Number of parallel worker processes (default is 9).
        bin_size : int, optional
            Target number of documents per bin (default is 1000).
        model : {"bm25", "tfidf"}, optional
            Embedding model type (default is "bm25").
        k : float, optional
            BM25 parameter k (default is 1.2).
        b : float, optional
            BM25 parameter b (default is 0.75).
        emb_seed : int, optional
            Random seed used for embedding hashing (default is 42).
        dimension : int, optional
            Dimension of the output embeddings (default is 300).
        """
        global CMS_DIM, CS_DIM, SEEDS

        cs_dim_temp = CS.compute_width_depth(epsilon, delta)
        cms_dim_temp = CMS.compute_width_depth(epsilon, delta)
        CS_DIM = cs_dim_temp
        CMS_DIM = cms_dim_temp
        SEEDS = {
            "cms": [random.randint(0, 2**32 - 1) for _ in range(cms_dim_temp[1])],
            "cs": [random.randint(0, 2**32 - 1) for _ in range(cs_dim_temp[1])],
            "cs_sign": [random.randint(0, 2**32 - 1) for _ in range(cs_dim_temp[1])]
        }

        self.src_path = src_path
        self.tmp_path = tmp_path
        self.bin_size = bin_size
        self.bins, self.doc_counter, self.doc_filenames = self._distribute_files_lpt()

        self.num_procs = num_procs
        self.sidf_cms_path = None
        self.sidf_cs_path = None

        self.model = model
        self.k = k
        self.b = b
        self.emb_seed = emb_seed
        self.dimension = dimension

        self.out_json = out_json
        self.out_npz = out_npz

        self.avg_dl: float = 1.0

    @staticmethod
    def _sketch_params_init(cms_dim: Tuple[int, int], cs_dim: Tuple[int, int], seeds: Dict[str, List[int]]) -> None:
        """
        Initializer for worker processes building local sketches.

        Parameters
        ----------
        cms_dim : Tuple[int, int]
            The CMS dimensions (width, depth).
        cs_dim : Tuple[int, int]
            The CS dimensions (width, depth).
        seeds : dict
            A dictionary with keys "cms", "cs", and "cs_sign" containing seeds.
        """
        global CMS_DIM, CS_DIM, SEEDS
        CMS_DIM = cms_dim
        CS_DIM = cs_dim
        SEEDS = seeds

    @staticmethod
    def _worker_build_local_cms(filepaths: List[str]) -> Tuple[CMS, CS, int]:
        """
        Build local CMS and CS sketches for a bin of files and compute the total token count.

        Note
        ----
        The document count is not accumulated here because it is already computed
        in the constructor (self.doc_counter).

        Parameters
        ----------
        filepaths : List[str]
            List of file paths to process.

        Returns
        -------
        Tuple[CMS, CS, int]
            A tuple containing:
              - local_cms : CMS instance
              - local_cs : CS instance
              - total_tokens : int (the sum of token counts over the processed files)
        """
        local_cms = CMS(rows=CMS_DIM[1], cols=CMS_DIM[0], seeds=SEEDS["cms"])
        local_cs = CS(rows=CS_DIM[1], cols=CS_DIM[0], seeds=SEEDS["cs"], sign_seeds=SEEDS["cs_sign"])
        token_sum = 0
        for fp in filepaths:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            tokens = [tok for tok in TOKEN_PATTERN.split(text.lower()) if tok]
            token_sum += len(tokens)
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                local_cms.update(tok, 1)
                local_cs.update(tok, 1)
        return local_cms, local_cs, token_sum

    def _distribute_files_lpt(self) -> Tuple[List[List[Tuple[int, str, int]]], int, List[str]]:
        """
        Distribute files into bins using Longest Processing Time (LPT) bin packing.

        Each bin is a list of tuples: (doc_index, file_path, file_size).

        Returns
        -------
        Tuple[List[List[Tuple[int, str, int]]], int, List[str]]
            - bins: List of bins (each bin is a list of document tuples).
            - total_docs: Total number of documents.
            - doc_filenames_ordered: List of file paths ordered by document index.
        """
        files = self._gather_files_list()  # List of (file_path, file_size)
        sorted_files = sorted(files, key=lambda x: x[1], reverse=True)
        doc_counter = len(sorted_files)
        k = max(1, math.ceil(doc_counter / self.bin_size))
        bins = [[] for _ in range(k)]
        bin_loads = [0] * k

        for doc_idx, (fp, size) in enumerate(sorted_files):
            idx = min(range(k), key=lambda i: bin_loads[i])
            bins[idx].append((doc_idx, fp, size))
            bin_loads[idx] += size

        all_docs = [doc for bin in bins for doc in bin]
        all_docs_sorted = sorted(all_docs, key=lambda x: x[0])
        doc_filenames_ordered = [fp for (_, fp, _) in all_docs_sorted]
        return bins, doc_counter, doc_filenames_ordered

    def _gather_files_list(self) -> List[Tuple[str, int]]:
        """
        Gather .txt files from the source directory.

        Returns
        -------
        List[Tuple[str, int]]
            A list of tuples containing (file_path, file_size).
        """
        out = []
        for fname in os.listdir(self.src_path):
            if fname.endswith(".txt"):
                fp = os.path.join(self.src_path, fname)
                if os.path.isfile(fp):
                    out.append((fp, os.path.getsize(fp)))
        return out

    def build_sidf(self) -> None:
        """
        Build global SIDF objects from local sketches and compute the average document length.

        The steps are:
          1. Prepare bins (each bin is a list of file paths).
          2. In parallel, build partial CMS and CS sketches and count token lengths.
          3. Merge partial sketches into global SIDF objects.
          4. Compute the average document length on the fly.
          5. Save SIDF objects to disk.

        Returns
        -------
        None
        """
        bins_for_sidf = [[fp for (_, fp, _) in bin] for bin in self.bins]

        total_tokens = 0
        local_cms_list = []
        local_cs_list = []
        with multiprocessing.Pool(processes=self.num_procs,
                                  initializer=IndexEmbeddingBuilder._sketch_params_init,
                                  initargs=(CMS_DIM, CS_DIM, SEEDS)) as pool:
            partial_results = pool.imap_unordered(IndexEmbeddingBuilder._worker_build_local_cms, bins_for_sidf)
            for cms_sketch, cs_sketch, token_sum in partial_results:
                local_cms_list.append(cms_sketch)
                local_cs_list.append(cs_sketch)
                total_tokens += token_sum

        if self.doc_counter > 0:
            self.avg_dl = total_tokens / self.doc_counter
        else:
            self.avg_dl = 1.0

        sidf_cms = SIDF(doc_count=self.doc_counter, sketches=local_cms_list)
        sidf_cs = SIDF(doc_count=self.doc_counter, sketches=local_cs_list)

        self.sidf_cms_path = os.path.join(self.tmp_path, "cms.msgpk")
        self.sidf_cs_path = os.path.join(self.tmp_path, "cs.msgpk")
        sidf_cms.save(self.tmp_path, "cms.msgpk")
        sidf_cs.save(self.tmp_path, "cs.msgpk")
        print(f"[build_sidf] Saved 'cms.msgpk' and 'cs.msgpk' to {self.tmp_path} with avg_dl={self.avg_dl}")

    @staticmethod
    def _sidf_init(sidf_cms_path: str, sidf_cs_path: str, num_docs_: int, avg_dl_: float,
                   emb_seed_: int, dimension_: int, model_: ModelCateg, k_: float, b_: float) -> None:
        """
        Initializer for worker processes building embeddings.

        Loads SIDF objects and sets embedding parameters as module-level globals.

        Parameters
        ----------
        sidf_cms_path : str
            Path to the CMS SIDF msgpack file.
        sidf_cs_path : str
            Path to the CS SIDF msgpack file.
        num_docs_ : int
            Total number of documents.
        avg_dl_ : float
            The average document length.
        emb_seed_ : int
            The embedding hash seed.
        dimension_ : int
            The embedding dimension.
        model_ : {"bm25", "tfidf"}
            The embedding model type.
        k_ : float
            BM25 parameter k.
        b_ : float
            BM25 parameter b.
        """
        global SIDF_CMS, SIDF_CS, NUM_DOCS, AVG_DL, EMB_SEED, DIMENSION, MODEL, K, B
        SIDF_CMS = SIDF.load(sidf_cms_path)
        SIDF_CS = SIDF.load(sidf_cs_path)
        NUM_DOCS = num_docs_
        AVG_DL = avg_dl_
        EMB_SEED = emb_seed_
        DIMENSION = dimension_
        MODEL = model_
        K = k_
        B = b_

    @staticmethod
    def _worker_build_embeddings(bin_data: List[Tuple[int, str, int]]) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Build embeddings for all documents in a bin.

        For each document, compute two embedding vectors (from CMS and CS SIDF objects)
        using either TF–IDF or BM25 weighting.

        Parameters
        ----------
        bin_data : List[Tuple[int, str, int]]
            List of tuples (doc_index, file_path, file_size).

        Returns
        -------
        List[Tuple[int, np.ndarray, np.ndarray]]
            A list of tuples containing (doc_index, cms_vec, cs_vec).
        """
        results = []
        for doc_idx, filepath, _ in bin_data:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            tokens = [tok for tok in TOKEN_PATTERN.split(text.lower()) if tok]
            tf_counts: Dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
            doc_length = len(tokens)
            cms_vec = np.zeros(DIMENSION, dtype=np.float32)
            cs_vec = np.zeros(DIMENSION, dtype=np.float32)
            for word, tf_val in tf_counts.items():
                if MODEL == "tfidf":
                    presence_cms = SIDF_CMS.query(word)
                    presence_cs = SIDF_CS.query(word)
                    idf_cms = math.log(NUM_DOCS / presence_cms) if presence_cms > 0 else 0.0
                    idf_cs = math.log(NUM_DOCS / presence_cs) if presence_cs > 0 else 0.0
                    score_cms = tf_val * idf_cms
                    score_cs = tf_val * idf_cs
                elif MODEL == "bm25":
                    idf_cms = SIDF_CMS[word]
                    idf_cs = SIDF_CS[word]
                    denom_cms = tf_val + K * (1 - B + B * (doc_length / AVG_DL))
                    denom_cs = tf_val + K * (1 - B + B * (doc_length / AVG_DL))
                    score_cms = idf_cms * (tf_val * (K + 1)) / (denom_cms if denom_cms > 0 else 1.0)
                    score_cs = idf_cs * (tf_val * (K + 1)) / (denom_cs if denom_cs > 0 else 1.0)
                else:
                    score_cms = score_cs = 0.0
                idx = mmh3.hash(word, EMB_SEED) % DIMENSION
                cms_vec[idx] += score_cms
                cs_vec[idx] += score_cs
            results.append((doc_idx, cms_vec, cs_vec))
        return results

    def build_embeddings_for_files(self) -> None:
        """
        Build embeddings for all documents and save the results.

        The process is:
          1. Use the precomputed average document length from build_sidf.
          2. Process each bin in parallel to compute embeddings.
          3. Save the ordered document filenames and embeddings to disk.

        Returns
        -------
        None
        """
        if not self.bins:
            print("[build_embeddings_for_files] No files found.")
            return

        cms_embs = np.zeros((self.doc_counter, self.dimension), dtype=np.float32)
        cs_embs = np.zeros((self.doc_counter, self.dimension), dtype=np.float32)

        pool_init_args = (self.sidf_cms_path, self.sidf_cs_path, self.doc_counter,
                          self.avg_dl, self.emb_seed, self.dimension, self.model, self.k, self.b)
        with multiprocessing.Pool(processes=self.num_procs,
                                  initializer=IndexEmbeddingBuilder._sidf_init,
                                  initargs=pool_init_args) as pool:
            results_iter = pool.imap_unordered(IndexEmbeddingBuilder._worker_build_embeddings, self.bins)
            for sublist in results_iter:
                for doc_idx, cms_vec, cs_vec in sublist:
                    cms_embs[doc_idx] = cms_vec
                    cs_embs[doc_idx] = cs_vec

        out_json_path = os.path.join(self.tmp_path, self.out_json)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_filenames, f, ensure_ascii=False, indent=2)
        out_npz_path = os.path.join(self.tmp_path, self.out_npz)
        np.savez_compressed(out_npz_path, cms_embs=cms_embs, cs_embs=cs_embs)
        print(f"[build_embeddings_for_files] Model={self.model} => wrote {self.out_json} & {self.out_npz}.")

    def build(self) -> None:
        """
        Run the full pipeline: build SIDF sketches and then compute document embeddings.

        Returns
        -------
        None
        """
        self.build_sidf()
        self.build_embeddings_for_files()

    @staticmethod
    def load_embeddings_and_filenames(embeddings_path: str, filenames_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load document embeddings from an NPZ file and corresponding filenames from JSON.

        Parameters
        ----------
        embeddings_path : str
            Path to the NPZ file containing embeddings.
        filenames_path : str
            Path to the JSON file with document filenames.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list]
            cms_embs, cs_embs, doc_filenames.
        """
        data = np.load(embeddings_path)
        cms_embs = data["cms_embs"]
        cs_embs = data["cs_embs"]
        with open(filenames_path, "r", encoding="utf-8") as f:
            doc_filenames = json.load(f)
        return cms_embs, cs_embs, doc_filenames

    @staticmethod
    def build_faiss_index(embs: np.ndarray) -> faiss.Index:
        """
        Build a Faiss L2 index (brute-force) from the given NxD embedding array.

        Parameters
        ----------
        embs : np.ndarray
            Embedding array of shape (num_docs, dim).

        Returns
        -------
        faiss.Index
            The Faiss L2 index.
        """
        num_docs, dim = embs.shape
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        return index

    @staticmethod
    def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
        """
        Normalize each embedding to unit norm.

        Parameters
        ----------
        embs : np.ndarray
            Embedding array of shape (num_docs, dim).

        Returns
        -------
        np.ndarray
            Normalized embeddings.
        """
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embs / norms

    @staticmethod
    def build_faiss_index_cosine(embs: np.ndarray) -> faiss.Index:
        """
        Build a Faiss index for cosine similarity using inner product.

        Assumes the embeddings are normalized.

        Parameters
        ----------
        embs : np.ndarray
            Normalized embedding array of shape (num_docs, dim).

        Returns
        -------
        faiss.Index
            The Faiss index based on inner product.
        """
        num_docs, dim = embs.shape
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        return index

    @staticmethod
    def find_similar_documents(index: faiss.Index, doc_id: int, embs: np.ndarray, doc_filenames: list, top_k: int = 5) -> None:
        """
        Find and print the top_k nearest neighbors for a document using the Faiss index.

        Parameters
        ----------
        index : faiss.Index
            The Faiss index built from the embeddings.
        doc_id : int
            The document ID to query.
        embs : np.ndarray
            The embedding array.
        doc_filenames : list
            List of document filenames.
        top_k : int, optional
            Number of neighbors to retrieve (default is 5).

        Returns
        -------
        None
        """
        query_vector = embs[doc_id].reshape(1, -1)
        distances, indices = index.search(query_vector, top_k)
        distances = distances[0]
        indices = indices[0]
        print(f"\n[Query doc_id={doc_id}, filename={doc_filenames[doc_id]}]")
        for rank, neighbor_id in enumerate(indices):
            dist = distances[rank]
            print(f"  Rank={rank} -> doc_id={neighbor_id}, filename='{doc_filenames[neighbor_id]}', dist={dist:.4f}")

    @staticmethod
    def find_farthest_documents(index: faiss.Index, doc_id: int, embs: np.ndarray, doc_filenames: list, tail_k: int = 5) -> None:
        """
        Find and print the tail_k farthest documents for a given document.

        Parameters
        ----------
        index : faiss.Index
            The Faiss index.
        doc_id : int
            The document ID to query.
        embs : np.ndarray
            The embedding array.
        doc_filenames : list
            List of document filenames.
        tail_k : int, optional
            Number of farthest documents to retrieve (default is 5).

        Returns
        -------
        None
        """
        num_docs = index.ntotal
        query_vector = embs[doc_id].reshape(1, -1)
        distances, indices = index.search(query_vector, num_docs)
        distances = distances[0]
        indices = indices[0]
        farthest_dists = distances[-tail_k:]
        farthest_ids = indices[-tail_k:]
        print(f"\n[Query doc_id={doc_id}, filename={doc_filenames[doc_id]}]")
        for rank_offset in range(tail_k):
            rank = tail_k - 1 - rank_offset
            dist = farthest_dists[rank]
            far_id = farthest_ids[rank]
            print(f"   Tail Rank {tail_k - rank} -> doc_id={far_id}, filename='{doc_filenames[far_id]}', dist={dist:.4f}")

    @staticmethod
    def compute_distance_correlation(embs1: np.ndarray, embs2: np.ndarray, metric: str = "l2", sample_size: int = None) -> float:
        """
        Compute the Pearson correlation coefficient between the distances computed
        on two embedding arrays using all document pairs.

        Parameters
        ----------
        embs1 : np.ndarray
            First embedding array of shape (num_docs, dim).
        embs2 : np.ndarray
            Second embedding array of shape (num_docs, dim).
        metric : str, optional
            Distance metric ("l2" or "cosine"). For "cosine", embeddings should be normalized.
            (default is "l2").
        sample_size : int, optional
            Number of random document pairs to sample. If None, all pairs are used.

        Returns
        -------
        float
            The Pearson correlation coefficient between the two distance vectors.
        """
        num_docs = embs1.shape[0]
        # If sample_size is None, use all pairs.
        if sample_size is None:
            # Total pairs: n*(n-1)/2 (this can be very large!)
            sample_size = num_docs * (num_docs - 1) // 2

        dists1 = []
        dists2 = []
        for _ in range(sample_size):
            i, j = random.sample(range(num_docs), 2)
            if metric == "l2":
                d1 = np.linalg.norm(embs1[i] - embs1[j])
                d2 = np.linalg.norm(embs2[i] - embs2[j])
            elif metric == "cosine":
                d1 = 1 - np.dot(embs1[i], embs1[j])
                d2 = 1 - np.dot(embs2[i], embs2[j])
            else:
                raise ValueError("Unsupported metric.")
            dists1.append(d1)
            dists2.append(d2)
        corr = np.corrcoef(dists1, dists2)[0, 1]
        return corr

    @staticmethod
    def run_isolation_forest(embs: np.ndarray, contamination: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run an Isolation Forest on the embeddings to detect outliers.

        Parameters
        ----------
        embs : np.ndarray
            Embedding array of shape (num_docs, dim).
        contamination : float, optional
            The proportion of outliers in the data (default is 0.05).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple (inlier_mask, predictions) where inlier_mask is a boolean array indicating inliers.
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(embs)  # 1 for inliers, -1 for outliers
        inlier_mask = predictions == 1
        return inlier_mask, predictions


if __name__ == "__main__":
    ############################################################################
    # Build or load embeddings using IndexEmbeddingBuilder
    ############################################################################
    src_path = r"F:\repos\proxi_pmi\src\tmp\txt"
    tmp_path = r"F:\repos\proxi_pmi\src\tmp"

    out_json = "doc_filenames.json"
    out_npz = "doc_embeddings.npz"

    # If the embeddings already exist, load them; otherwise build them.
    npz_path = os.path.join(tmp_path, out_npz)
    json_path = os.path.join(tmp_path, out_json)
    if os.path.exists(npz_path) and os.path.exists(json_path):
        print("Embeddings found. Loading saved embeddings and filenames...")
    else:
        builder = IndexEmbeddingBuilder(
            src_path=src_path,
            tmp_path=tmp_path,
            out_json=out_json,
            out_npz=out_npz,
            epsilon=0.01,
            delta=0.01,
            num_procs=9,           # Adjust as needed
            bin_size=10000,
            model="bm25",
            k=1.2,
            b=0.75,
            emb_seed=42,
            dimension=300
        )
        builder.build()

    cms_embs, cs_embs, doc_filenames = IndexEmbeddingBuilder.load_embeddings_and_filenames(npz_path, json_path)
    print("CMS Embeddings shape:", cms_embs.shape)
    print("CS Embeddings shape:", cs_embs.shape)
    print(f"Number of doc filenames: {len(doc_filenames)}")

    ############################################################################
    # Run Isolation Forest outlier detection on CMS embeddings
    ############################################################################
    inlier_mask, predictions = IndexEmbeddingBuilder.run_isolation_forest(cms_embs, contamination=0.05)
    num_inliers = np.sum(inlier_mask)
    num_outliers = len(inlier_mask) - num_inliers
    print(f"Isolation Forest: Detected {num_outliers} outliers out of {len(inlier_mask)} documents.")

    # Filter embeddings and filenames
    cms_embs_filtered = cms_embs[inlier_mask]
    cs_embs_filtered = cs_embs[inlier_mask]
    doc_filenames_filtered = [doc_filenames[i] for i in range(len(doc_filenames)) if inlier_mask[i]]

    # Save filtered embeddings and filenames
    filtered_npz_path = os.path.join(tmp_path, "doc_embeddings_filtered.npz")
    filtered_json_path = os.path.join(tmp_path, "doc_filenames_filtered.json")
    np.savez_compressed(filtered_npz_path, cms_embs=cms_embs_filtered, cs_embs=cs_embs_filtered)
    with open(filtered_json_path, "w", encoding="utf-8") as f:
        json.dump(doc_filenames_filtered, f, ensure_ascii=False, indent=2)
    print(f"Filtered embeddings and filenames saved: {filtered_npz_path}, {filtered_json_path}")

    ############################################################################
    # Build Faiss L2 indexes for both CMS and CS embeddings (using filtered data)
    ############################################################################
    l2_index_cms = IndexEmbeddingBuilder.build_faiss_index(cms_embs_filtered)
    l2_index_cs = IndexEmbeddingBuilder.build_faiss_index(cs_embs_filtered)

    ############################################################################
    # Build cosine similarity indexes (normalize embeddings first)
    ############################################################################
    cms_embs_norm = IndexEmbeddingBuilder.normalize_embeddings(cms_embs_filtered)
    cs_embs_norm = IndexEmbeddingBuilder.normalize_embeddings(cs_embs_filtered)
    cosine_index_cms = IndexEmbeddingBuilder.build_faiss_index_cosine(cms_embs_norm)
    cosine_index_cs = IndexEmbeddingBuilder.build_faiss_index_cosine(cs_embs_norm)

    ############################################################################
    # Query similar and farthest documents for selected document IDs (filtered set)
    ############################################################################
    doc_ids_to_query = [0, 5]
    for doc_id in doc_ids_to_query:
        if 0 <= doc_id < len(doc_filenames_filtered):
            print("L2 similarity (FIASS build in feature)")
            print("CMS")
            IndexEmbeddingBuilder.find_similar_documents(l2_index_cms, doc_id, cms_embs_filtered, doc_filenames_filtered, top_k=5)
            IndexEmbeddingBuilder.find_farthest_documents(l2_index_cms, doc_id, cms_embs_filtered, doc_filenames_filtered, tail_k=5)
            print("CS")
            IndexEmbeddingBuilder.find_similar_documents(l2_index_cs, doc_id, cs_embs_filtered, doc_filenames_filtered, top_k=5)
            IndexEmbeddingBuilder.find_farthest_documents(l2_index_cs, doc_id, cs_embs_filtered, doc_filenames_filtered, tail_k=5)

            print("\nCosine similarity (using inner product) queries:")
            print("CMS")
            IndexEmbeddingBuilder.find_similar_documents(cosine_index_cms, doc_id, cms_embs_norm, doc_filenames_filtered, top_k=5)
            IndexEmbeddingBuilder.find_farthest_documents(cosine_index_cms, doc_id, cms_embs_norm, doc_filenames_filtered, tail_k=5)
            print("CS")
            IndexEmbeddingBuilder.find_similar_documents(cosine_index_cs, doc_id, cs_embs_norm, doc_filenames_filtered, top_k=5)
            IndexEmbeddingBuilder.find_farthest_documents(cosine_index_cs, doc_id, cs_embs_norm, doc_filenames_filtered, tail_k=5)
        else:
            print(f"Invalid doc_id={doc_id}. Skipping.")

    ############################################################################
    # Compute correlation between distances from CMS and CS embeddings (filtered)
    ############################################################################
    corr_l2 = IndexEmbeddingBuilder.compute_distance_correlation(cms_embs_filtered, cs_embs_filtered, metric="l2", sample_size=1000)
    corr_cosine = IndexEmbeddingBuilder.compute_distance_correlation(cms_embs_norm, cs_embs_norm, metric="cosine", sample_size=1000)
    print("\nCorrelation between L2 distances (CMS vs CS):", corr_l2)
    print("Correlation between cosine distances (CMS vs CS):", corr_cosine)
