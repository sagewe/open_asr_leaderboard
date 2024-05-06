import argparse

import os
import shutil
import torch
import evaluate
import soundfile

from tqdm import tqdm
from normalizer import data_utils
import numpy as np

from nemo.collections.asr.models import ASRModel
import time

DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")

wer_metric = evaluate.load("wer")


def dataset_iterator(dataset):
    for i, item in enumerate(dataset):
        # import ipdb; ipdb.set_trace()
        yield {
            **item["audio"],
            "reference": item["norm_text"],
        }


def write_audio(buffer, cache_prefix) -> list:
    cache_dir = os.path.join(DATA_CACHE_DIR, cache_prefix)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir)

    data_paths = []
    for idx, data in enumerate(buffer):
        fn = os.path.basename(data['audio_filename'])
        fn = os.path.splitext(fn)[0]
        path = os.path.join(cache_dir, f"{idx}_{fn}.wav")
        data_paths.append(path)

        soundfile.write(path, data["array"], samplerate=data['sample_rate'])

    return data_paths


def pack_results(results: list, references, transcriptions):
    for sample, transcript in zip(references, transcriptions):
        result = {'reference': sample, 'pred_text': transcript}
        results.append(result)
    return results


def main(args):

    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(args.model_id, map_location=device)  # type: ASRModel
    asr_model.freeze()

    dataset = data_utils.load_data(args)

    def benchmark_batch(batch):

        # get audio stats
        audio = [np.float32(sample["array"]) for sample in batch["audio"]]
        batch["audio_length"] = [len(sample) / 16_000 for sample in audio]
        minibatch_size = len(audio)

        # timing step
        start_time = time.time()
        transcriptions = asr_model.transcribe(audio, batch_size=minibatch_size, verbose=False)
        
        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time"] = minibatch_size * [(time.time() - start_time) / minibatch_size]

        if type(transcriptions) == tuple and len(transcriptions) == 2:
                transcriptions = transcriptions[0]
        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in transcriptions]
        batch["references"] = batch["norm_text"]

        return batch

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(benchmark_batch, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    all_results = {
        "audio_length": [],
        "transcription_time": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples"):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length"],
        transcription_time=all_results["transcription_time"],
    )

    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=all_results['references'], predictions=all_results['predictions'])
    wer = round(100 * wer, 2)

    transcription_time = sum(all_results["transcription_time"])
    audio_length = sum(all_results["audio_length"])
    rtfx = audio_length / transcription_time
    rtfx = round(rtfx, 2)

    print("RTFX:", rtfx)
    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        '--dataset_path', type=str, default='esb/datasets', help='Dataset path. By default, it is `esb/datasets`'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
