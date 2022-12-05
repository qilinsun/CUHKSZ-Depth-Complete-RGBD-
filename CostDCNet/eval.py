from options import Options
from mp_trainer import Mp_trainer
options = Options()
opts = options.parse()

if __name__ == "__main__":
    print('Training mode')
    trainer = Mp_trainer(opts)
    trainer.evaluate()
