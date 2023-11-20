from pylsl import StreamInlet, resolve_stream
from time import time, sleep
import numpy as np


def main():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EOG')

    inlet = StreamInlet(streams[0])

    # IDEAL: tamanho do protocolo = 28, com movimento aleatórios
    protocol = ['CORREÇÃO', 'dir', 'esq', 'cima', 'baixo', 'cima', 'baixo',
                'baixo', 'esq', 'dir', 'baixo', 'dir', 'dir', 'esq', 'cima',
                'baixo', 'cima', 'esq', 'dir', 'cima', 'esq', 'baixo', 'esq',
                'dir', 'esq', 'cima', 'dir', 'cima', 'baixo']
    
    sample, _ = inlet.pull_sample()

    trial = list()
    for prot in protocol:
        print(f"MOVIMENTO >>> {prot.upper()} <<<")
        start_t = time()
        mov = list()
        while (time() - start_t) < T_MOV:
            sample, _ = inlet.pull_sample()
            mov.append(sample)
        # print(f"Quantidade de pontos: {len(mov)}")
        trial.append(mov[:(FS * T_MOV)])
        print("\nDescanço!\n")
        sleep(T_REST)
    trial = np.array(trial[1:])
    print(f"\nDimensionalidade: {trial.shape}")


if __name__ == '__main__':
    T_REST = 3
    T_MOV = 5
    FS = 200
    main()
