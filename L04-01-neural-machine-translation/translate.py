import os
from run import beam_search
from nmt_model import Hypothesis, NMT

def do_translate(src_sentence: str) -> str:

    hypotheses = beam_search(model, src_sentence,
                                beam_size=5,
                                max_decoding_time_step=70)

    tgt_sentence = []

    for src_sent, hyps in zip(src_sentence, hypotheses):
        top_hyp = hyps[0]
        hyp_sent = ' '.join(top_hyp.value)
        tgt_sentence.append(hyp_sent + '\n')

    return ''.join(tgt_sentence)

if __name__ == "__main__":
    os.system("clear")
    model = NMT.load("./training_results/model.bin")

    src_sentence = ""

    while src_sentence != "<salir>":
        src_sentence = input("Escribe la oración en Español: ")

        if src_sentence != "<salir>" and len(src_sentence.split()) > 0:
            src_sentence = [src_sentence.split()]

            result = do_translate(src_sentence)

            print(f"Traducción: {result}")