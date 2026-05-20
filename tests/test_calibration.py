# Autor: Massanori
# Data: 20/05/2026
# Descricao: ARQUIVO DEPRECATED. Os testes deste arquivo foram consolidados
#            em tests/test_conformal.py + tests/test_coverage.py durante o
#            commit ae897c3 (infra de calibracao do S5.7), que reescreveu
#            a API com nomes mais descritivos:
#                apply_qhat_qr      -> apply_cqr_interval
#                apply_qhat_resm    -> apply_resm_interval
#                nonconformity_qr   -> cqr_score
#                nonconformity_resm -> scaled_cp_score
#                compute_qhat       -> conformal_quantile
#                calibrate(kind=)   -> calibrate_qr / calibrate_resm (separados)
#                evaluate           -> aplicar interval + empirical_coverage
#                coverage_stats     -> empirical_coverage (em src/metrics/coverage.py)
#            Os 3 testes unicos valiosos deste arquivo (convergencia para
#            Phi^-1(0.9) com N(0,1), correcao finite-sample para n pequeno,
#            q_hat negativo para intervalo conservador demais) foram
#            migrados para tests/test_conformal.py no mesmo commit deste
#            placeholder. Este arquivo permanece sem funcoes test_* para
#            que pytest colete 0 testes silenciosamente, sem ImportError.


"""DEPRECATED.

Veja tests/test_conformal.py (16 + 3 testes migrados) e
tests/test_coverage.py (12 testes) para os testes da camada de
calibracao conforme.

A API antiga (apply_qhat_qr, compute_qhat, nonconformity_qr, etc.)
foi substituida pela API atual em src/calibration/ no commit ae897c3.
Este arquivo nao contem mais testes — pytest coleta 0 deste arquivo.
"""
