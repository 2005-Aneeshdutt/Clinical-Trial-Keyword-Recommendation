from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester
from ml.data_set_splitter import DataSetSplitter
import pickle
import sys
from pathlib import Path

def _find_latest_pickle(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob('model*.pickle'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def main():
    out_dir = Path('./output')
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional arg: explicit model path
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_pickle(out_dir)
    if model_path is None or not model_path.exists():
        raise SystemExit("No model pickle found in ./output. Train a model first or pass a path.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # print persisted metrics if present
    if hasattr(model, 'mse_test'):
        print(model.mse_test)
    if hasattr(model, 'apk_test'):
        print(model.apk_test)

    sds_path = Path('./output/fullAllPublicXML.sds')
    TEST = DataSetSplitter.get_test_utility_matrix(str(sds_path))

    mset = MseTester(model)
    mse = mset(TEST)
    print(mse)
    apkt = AveragePrecisionKTester(model)
    apk = apkt(TEST)
    print(apk)

if __name__ == '__main__':
    main()
