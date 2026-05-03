from ..trainer.algs.MEND import MEND
from ..trainer.algs.SERAC import SERAC_MULTI
from ..trainer.algs.FT import FT
from ..trainer.algs.OURS import OURS
from ..trainer.algs.WISE import WISEMultimodal


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC_MULTI,
    'SERAC_MULTI': SERAC_MULTI,
    'FT': FT,
    'ft': FT,
    'lora': FT,
    'LORA': FT,
    'OURS': OURS,
    'WISE': WISEMultimodal
}
