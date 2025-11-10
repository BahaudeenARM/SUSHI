from tracker_frameworks.SUSHI.scripts.src.utils.deterministic import make_deterministic
from tracker_frameworks.SUSHI.scripts.src.tracker.hicl_tracker import HICLTracker
from tracker_frameworks.SUSHI.scripts.src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
from configs.config import get_arguments

def process_detections_with_SUSHI(config_list, output_folder_path):
    config = get_arguments(config_list)
    seqs, splits = get_seqs_from_splits(data_path=config.data_path, test_split=config.test_splits[0])

    # Initialize the tracker
    hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

    # Load the pretrained model
    hicl_tracker.model = hicl_tracker.load_pretrained_model()

    # Track
    hicl_tracker.track(dataset=hicl_tracker.test_dataset, output_path=output_folder_path,
                                                                    mode='test',
                                                                    oracle=False)