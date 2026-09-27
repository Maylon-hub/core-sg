# Authorship approval required

Proposal for core-sg-mustache 0.4.5rc3; confirm with Murilo/MIDAS, Maylon and
the original Python maintainers before software publication.

| Role | Evidence / attribution | Decision still required |
|---|---|---|
| Original scientific work | Antonio Cavalcante Araujo Neto, Murilo Coelho Naldi, Ricardo J. G. B. Campello, Jorg Sander; CORE-SG paper, ICDE 2022 | Preserve the scientific reference; it does not assign implementation authorship |
| Original Python implementation | Existing metadata/copyright: Midas Core-SG Team; Git records Gabriel Orlando / Gab0410 | Confirm individual identity/aliases and contributor list with maintainers; paper authors are not automatically Python authors |
| Later contributions | Git records Maylon Martins de Melo / Maylon and upstream contributors | Confirm identities, contribution descriptions and software citation order; commit counts are not author ranking |
| Integration/modernization in this project | Maylon's MustaCHE integration, full k_max support preservation, repeated extraction/distance compatibility fixes, tests and release qualification | Confirm acknowledgement/authorship wording and distinction from canonical upstream releases |
| Institutional supervision/ownership | Murilo is the stated MustaCHE IC advisor; upstream repository belongs to MIDAS | Confirm maintainer, institutional owner and canonical release policy; no transfer or institutional appointment is implied |

## CITATION.cff fields requiring confirmation

- `authors`: provisional team plus Maylon; confirm collective name, individual
  names/aliases, list completeness and order.
- `title`: `CORE-SG MustaCHE integration fork`; confirm its distinction from
  canonical upstream software and the scientific article.
- `message`: provisional approval notice; change after decisions are recorded.
- `repository-code`: personal integration fork; any MIDAS destination requires
  an authorized ownership/integration decision.
- Future `orcid`, `affiliation`, `doi`, `date-released`: absent; add only verified
  values after the corresponding human decision or actual release/deposit.

`version: 0.4.5rc3`, software type, license and the original article reference
are release/citation facts. Existing copyright and HDBSCAN third-party notices
must remain. No individual name expansion, affiliation, ORCID or author ranking
is invented here. Approval is required for publication, not for CI execution.
