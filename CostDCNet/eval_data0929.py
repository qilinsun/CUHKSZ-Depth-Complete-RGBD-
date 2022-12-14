from data0929_evaler import Data0929_trainer
from options import Options

options = Options()
opts = options.parse()

if __name__ == "__main__":
    print('Training mode')
    trainer = Data0929_trainer(opts)
    trainer.evaluate()

