from src.grounding import is_grounded, locate, normalize


def test_tolerates_pdf_hyphenation():
    src = "we therefore allow the appeal and dismiss the plain- tiff's suit."
    assert is_grounded("We therefore allow the appeal and dismiss the plaintiff's suit.", src)


def test_tolerates_mid_sentence_line_breaks():
    src = "The plaintiff\nfiled a suit for\ndeclaration of title."
    assert is_grounded("The plaintiff filed a suit for declaration of title.", src)


def test_accepts_short_real_disposition():
    assert is_grounded("Appeal dismissed.", "...with costs. Appeal dismissed. Agent for")


def test_refuses_fabrication():
    assert not is_grounded("We accordingly allow the appeal.", "the appeal is dismissed with costs")


def test_refuses_ellipsis_reconstruction():
    assert not is_grounded("the appeal is ... allowed", "the appeal is hereby allowed")


def test_refuses_too_short():
    assert not is_grounded("allowed", "the appeal is allowed")


def test_locate_returns_offsets_that_index_the_original():
    src = "Preamble text. The plaintiff filed a suit for possession. Later text."
    span = locate("The plaintiff filed a suit for possession.", src)
    assert span is not None
    assert src[span[0]:span[1]] == "The plaintiff filed a suit for possession."


def test_locate_maps_back_through_hyphenation():
    src = "Earlier words. the plain- tiff filed a suit for posses- sion here. After."
    span = locate("the plaintiff filed a suit for possession here.", src)
    assert span is not None
    got = src[span[0]:span[1]]
    assert normalize(got) == normalize("the plaintiff filed a suit for possession here.")


def test_locate_returns_none_when_absent():
    assert locate("a sentence that is not there at all", "some other text entirely") is None
