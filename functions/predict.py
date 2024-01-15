from pathlib import Path
import main
import model
import preprocessing
import numpy as np

def run(args):
    main.prerun(args, run_dir=False, exp=False)

    split = args["split_dir"]
    pred_indices = np.loadtxt(Path(split, "test.txt"), dtype=int)

    data = preprocessing.load_hdf5_to_memory(args, None)
    ds, ds_len = preprocessing.load_dataset(data, pred_indices, None, args, type="pred")

    m = model.load_model(args)
    
    preds = m.predict(
        ds,
        steps = int(np.ceil(ds_len / args["batch_size"])),
    )

    path=args["run_dir"]+"predictions.npy"
    
    np.save(path, preds)
