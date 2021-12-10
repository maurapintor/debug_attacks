import numpy as np
import requests


def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))


def sampling_n_sphere(x, eps: float, p=np.inf):
    c = np.random.uniform(low=-1, high=1, size=x.shape).ravel()
    c = c / np.linalg.norm(c, ord=p) * eps
    return x + c


class CMetricScoreDifference:
    """Computes the target score, the competing score,
    and the difference between the two.
    """

    @classmethod
    def score_difference(cls, scores, y0, y_t=None):
        """
        Parameters
        ----------
        scores: np.array
            Array containing the scores along the path.
        y0: np.array
            Original class.
        y_t: int
            Target class. None if the attack is untargeted.
        Returns
        -------
        The target, competing and diff score for the attack.
        If the attack is untargeted, the target score will be the
        maximum score, excluding the original class, and the competing
        will be the score of the original class.
        If the attack is targeted, the target score will be the
        target class score, and the competing will be the maximum score,
        excluding the score of the target class.
        """
        rows = range(scores.shape[0])
        if y_t not in [None, False]:
            # targeted attack
            score_maximize = scores[rows, y_t].ravel()
            score_minimize = cls.competing_score(scores, exclude=y_t)
        else:
            score_maximize = cls.competing_score(scores, exclude=y0)
            score_minimize = scores[rows, y0].ravel()
        score_diff = score_minimize - score_maximize
        return score_maximize.ravel(), score_minimize.ravel(), \
               score_diff.ravel()

    @classmethod
    def competing_score(cls, logits, exclude):
        """Computes the best score out of all classes,
        excluding the one indicated. If no class is indicated,
        the excluded score will be the top score in the first
        row (initial prediction).
        Parameters
        ----------
        logits : CArray
            Outputs scores of the classifier.
        exclude: int
            Index of the score to exclude from the top-1
            extraction.
        Returns
        -------
        The competing score for the attack. If the attack is
        untargeted, the competing score will be the top score
        exluding the one that is currently max.
        """
        other_logits = np.copy(logits)
        rows = range(other_logits.shape[0])
        other_logits[rows, exclude] = -np.inf
        return other_logits.max(axis=1).ravel()
