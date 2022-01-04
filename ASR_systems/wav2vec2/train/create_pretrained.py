from transformers import Wav2Vec2ForCTC
import argparse
from os import listdir as ls
import models

def main(args):
    wav2vec = Wav2Vec2ForCTC.from_pretrained(args.pretrained if args.pretrained in ls() else f'facebook/{args.pretrained}')
    model = models.ASR_CTC.load_from_checkpoint(args.checkpoint, model=wav2vec, config=args)
    model.model.save_pretrained(args.output)
    print(f'--- Saved pre-trained model ---\n saved as: {args.output}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="check3.ckpt")
    parser.add_argument('--pretrained', type=str, default='wav2vec2-large-robust-ft-libri-960h')
    parser.add_argument('--output', type=str, default='wav2vec2_openslr_3')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()
    main(args)
