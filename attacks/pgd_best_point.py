from secml.adv.attacks import CFoolboxPGDLinf
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor


class CFoolboxPGDBest(CFoolboxPGDLinf):
    __class_type = 'e-foolbox-pgd-patched'

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxPGDLinf, self)._run(x, y, x_init)
        best_pt = self.x_seq[self.objective_function(self.x_seq).argmin(), :]
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(best_pt)

        return best_pt, f_opt
