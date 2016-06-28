import frog
import pickle
import tqdm
import glob

OUTPUT_FOLDER = '../data/frog_output/'



def frog_event(frogger, filename):
    id = filename.split('/')[-1].replace('.txt','')
    with open(filename, 'r') as f:
        text = f.read()
        output = frogger.process_raw(text)

    with open(OUTPUT_FOLDER+id+'.frog.out', 'w') as f:
            f.write(output)

if __name__ == '__main__':

    frogger = frog.Frog(frog.FrogOptions(parser=False,mwu=False,ner=False,morph=False,chunking=False, numThreads=4),'/etc/frog/frog.cfg')
    filenames  = glob.glob('../data/events/*.txt')

    for event in tqdm.tqdm(filenames):
        frog_event(frogger, event)