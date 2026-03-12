# Testes unitários do Core-SG

Esta pasta contém a suíte de testes em `pytest` para a biblioteca `core-sg`.

## Estrutura

- `conftest.py`: fixtures compartilhadas e ambiente fake de `hdbscan` para testes unitários puros.
- `helpers.py`: utilitários de teste, payloads fake e validadores.
- `test_class_initialization.py`: estado inicial, filtragem de kwargs e erros antes do `fit`.
- `test_class_fit.py`: `fit`, cache de artefatos, retorno de MST e objetos salvos do `fit`.
- `test_class_hierarchy.py`: extração de hierarquia, atualização de atributos e wrappers.
- `test_low_level_helpers.py`: KNN, merge de arestas, reweight e Kruskal.
- `test_reference_equivalence.py`: testes de integração com `hdbscan` real, com `skip` automático se a dependência não estiver instalada.
- `test_validate_class_core_sg.py`: teste para verificar funcionamento da classe do CoreSG
- `test_validate_class_core_sg_weight.py`: teste para verificar funcionamento da classe do CoreSG em relaçõ aos pesos MRD
- `test_validate_class_mst_weight.py`: Teste apenas para verificar funcionamento
- `test_validate_class.py`: Verificar funcionamento classe CoreSG
- `test_validate_core_sg_mst_weight.py`: Verificação de funções
- `test_validate_core_sg.py`: Verificação de funções
- `test_validate_core_sg_weight.py`: Verificação de funções
- `test_validate_mst_weight.py`: Verificação de funções


## Como executar

```bash
pytest teste -q
```

Para rodar apenas os testes unitários puros:

```bash
pytest teste/test_class_initialization.py teste/test_class_fit.py teste/test_class_hierarchy.py teste/test_low_level_helpers.py -q
```

Para rodar testes no modo debug

```bash
pytest tests -v -ra
```



