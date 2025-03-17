import torch
from torch.utils.data import Dataset, DataLoader, Subset
from bagz import BagDataSource
from config import data_config
from apache_beam import coders
import utils
import numpy as np
import tokenizer
import abc

dataloader_instance = None  # This will hold the created DataLoader instance

# to read from the bagz files
CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))


def _process_fen(fen: str) -> np.ndarray:
    return tokenizer.tokenize(fen).astype(np.int32)


def _process_move(move: str) -> np.ndarray:
    return np.asarray([utils.MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
    return utils.compute_return_buckets_from_returns(
        returns=np.asarray([win_prob]),
        bins_edges=return_buckets_edges,
    )


class ConvertToSequence(abc.ABC):
    """Base class for converting chess data to a sequence of integers."""

    def __init__(self, num_return_buckets: int) -> None:

        self._return_buckets_edges, _ = utils.get_uniform_buckets_edges_values(
            num_return_buckets,
        )
        # The loss mask ensures that we only train on the return bucket.
        self._loss_mask = np.full(
            shape=(self._sequence_length,),
            fill_value=True,
            dtype=bool,
        )
        self._loss_mask[-1] = False

    @property
    @abc.abstractmethod
    def _sequence_length(self) -> int:
        raise NotImplementedError()


class ConvertActionValueDataToSequence(ConvertToSequence):
    """Converts the fen, move, and win probability into a sequence of integers."""

    @property
    def _sequence_length(self) -> int:
        return tokenizer.SEQUENCE_LENGTH + 2  # (s) + (a) + (r)

    def map(
        self, element: bytes
    ):
        fen, move, win_prob = CODERS['action_value'].decode(element)
        state = _process_fen(fen)
        action = _process_move(move)
        return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
        sequence = np.concatenate([state, action, return_bucket])
        return torch.from_numpy(sequence), torch.from_numpy(self._loss_mask)


class ChessDataset(Dataset):
    def __init__(self, filename):
        self.bagdatasource = BagDataSource(filename)
        self.transform = ConvertActionValueDataToSequence(
            num_return_buckets=data_config.num_return_buckets)

    def __len__(self):
        return self.bagdatasource.__len__()

    def __getitem__(self, idx):
        # Return a single data point (e.g., a tuple of input and label)
        bytes = self.bagdatasource.__getitem__(idx)
        output_tensor = self.transform.map(bytes)
        return output_tensor


def create_dataloader():

    dataset = ChessDataset(data_config.filename)
    if (data_config.miniDataSet):
        dataset = Subset(dataset, list(range(data_config.mini_set_count)))

    global dataset_instance
    dataloader_instance = DataLoader(
        dataset, batch_size=data_config.batch_size, shuffle=data_config.shuffle)

    return dataloader_instance


# Creating the DataLoader instance for direct import
dataloader_instance = create_dataloader()

if (__name__ == "__main__"):

    for count, i in enumerate(dataloader_instance):
        if count == 0:
            print(i)
        print(type(i))
        print(len(i))
        if count == 100:
            break
