from src.authorities.normalize import extract


def labels(t):
    return [c.label for c in extract(t)]


def test_roman_order_rule():
    assert labels("an application under Order XXI Rule 90 of the Code of Civil Procedure") == ["CPC:O21R90"]


def test_digit_order_rule_does_not_collide_with_sections():
    got = labels("Order 21 Rule 97 CPC and Section 21 of the Code of Civil Procedure")
    assert "CPC:O21R97" in got and "CPC:21" in got and "CPC:97" not in got


def test_order_alone():
    assert labels("Order XXXIX of the Code of Civil Procedure") == ["CPC:O39"]


def test_act_named_before_the_provision():
    assert labels("Limitation Act, 1963, Article 65") == ["LIM:Art65"]
    assert labels("Transfer of Property Act, 1882, Section 53A") == ["TPA:53A"]


def test_unchanged_forms():
    assert labels("Section 53A of the Transfer of Property Act") == ["TPA:53A"]
    assert labels("Article 300A of the Constitution") == ["CONST:Art300A"]
    assert labels("Section 16(c) of the Specific Relief Act") == ["SRA:16(c)"]


def test_article_without_act_is_still_constitution():
    assert labels("a violation of Article 14") == ["CONST:Art14"]


def test_not_an_order_number():
    # "order dated" / an order number outside the CPC's 51 Orders must not become a CPC label
    assert not any(l.startswith("CPC:O") for l in labels("by order 1234 of the Collector"))


def test_rules_listed_after_an_order_inherit_it():
    got = labels("an application under Order XXII Rule 3 and Rule 4 of the Code")
    assert "CPC:O22R3" in got and "CPC:O22R4" in got and "CPC:R4" not in got
    got = labels("Order 22, Rules 3 and 4 CPC")
    assert "CPC:O22R3" in got and "CPC:O22R4" in got


def test_bare_rule_is_not_guessed_into_cpc():
    assert not any(l.startswith("CPC:") for l in labels("because Rule 3 will have to be read in the light of the schedule"))
