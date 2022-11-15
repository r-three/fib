import re

from src.data.data_preprocess import tokenize_prompted_input_text, does_tokenizer_addBosEosTokens
from transformers import AutoTokenizer


from src.data.templates import SUMMARIZATION_PROMPT_TEMPLATES
from src.utils.test_helpers import check_string_ends_with_another, check_string_starts_with_another, check_string_subset_of_another, check_string_equality

SHORT_DATAPOINT = {
    "input": "Forbes said Vergara's role as Gloria in Modern Family and some lucrative product endorsements helped her earn $43m (\u00a332.6m) in the last 12 months.\n It marks the fifth year the Colombian-American actress has topped the chart.\n Forbes also said she earned more than any of her male counterparts in the past year.\n The Big Bang Theory's Kaley Cuoco was the second-highest paid actress, earning $24.5m (\u00a318.6m).\n Cuoco tied with Vergara at the top of last year's Forbes list, when both actresses earned $28.5m (\u00a321.6m).\n The Mindy Project's Mindy Kaling is the biggest climber in this year's chart. Her earnings of $15m (\u00a311.4m) helped her to rise from eighth place in 2015 to third this year.\n Mariska Hargitay, who appears in Law & Order: Special Victims Unit, and Grey's Anatomy star Ellen Pompeo rounded off the top five.\n Source: Forbes\n This year's highest new entry on the Forbes list was Priyanka Chopra, who appears in ABC drama Quantico. She was the eighth highest earner with $11m (\u00a38.4m).\n Chopra, who is well known in India, is set to become more familiar to western audiences next year when she stars in Baywatch alongside Dwayne Johnson - the world's highest paid actor.\n Scandal star Kerry Washington, Stana Katic from Castle, The Good Wife's Julianna Margulies and Vergara's Modern Family co-star Julie Bowen also featured in this year's top 10.\n Follow us on Twitter @BBCNewsEnts, on Instagram, or if you have a story suggestion email entertainment.news@bbc.co.uk.",
    "choice": "Modern Family star Sofia Vergara has retained her title as the highest paid actress on US television, according to the latest Forbes magazine rich list."
}

LONG_DATAPOINT = {
    "input": "Archery, fencing, weightlifting and wheelchair rugby have also missed out.\n Cycling - which brought Team GB 12 medals in Rio - has had its funding cut by more than \u00a34m to \u00a325.98m.\n Badminton England chief executive Adrian Christy said he was \"staggered\" by the \"incomprehensible\" decision to remove the sport's funding.\n A total of \u00a3345m will be invested in 31 Olympic and Paralympic sports - \u00a32m less than the record \u00a3347m allocated for the Rio Games.\n As a result, UK Sport has set Team GB a target of winning 51-85 Olympic medals, and 115-162 Paralympic medals in Tokyo.\n Britain enjoyed unprecedented success at Rio 2016, with the Olympics yielding 67 medals and the Paralympics 147.\n Chair of UK Sport Rod Carr said the government, which provides funding alongside National Lottery money, has \"confirmed its commitment\" for Tokyo 2020.\n He added: \"These are critical funding decisions for sports to take them on their journey to Tokyo 2020 and beyond so the historic success at Rio can be maintained.\"\n Badminton, which was set a target of winning a medal in Rio, is the only sport that earned a podium place in the summer to have its funding removed.\n Marcus Ellis and Chris Langridge took bronze in the men's doubles after the sport was given \u00a35.74m in the last cycle.\n Christy said the decision represents a \"catastrophic impact on the sport\" and Badminton England would \"fight for the hopes and dreams\" of its players.\n \"How can you return from the best Games for more than a decade, in a year where our players have demonstrated world-class performances and where we can demonstrate the journey to Tokyo is on track, only be to have every penny of investment withdrawn?\" he said.\n \"What have we done wrong?\" added GB Badminton's performance director Jon Austin.\n Judo, which was given the same target as badminton and also claimed one bronze medal, has had its funding increased slightly.\n Liz Nicholl, CEO of UK Sport, said the decision to cut funding was not taken lightly.\n \"We would like to invest in every sport but the reality is we have to prioritise to protect and enhance the medal potential,\" she said.\n \"If we under-invest across the board then the British teams will ultimately underperform at the Games and medal success will be put at risk.\"\n Sports minister Tracey Crouch added: \"UK Sport's approach to elite sport has proven successful in Beijing, London and Rio and the ambition to win more medals in Tokyo is a bold one that, if achieved, would mean a sensational summer of sport in 2020.\"\n Basketball had its funding withdrawn in 2014 - and handball and volleyball lost theirs in 2012 - but say a UK Sport review last year to build \"performance pathways for future success\" was supposed to be aimed at such sports.\n A British Basketball statement, in conjunction with volleyball and handball, said: \"It appears that UK Sport has no interest in team sports and in particular refuses to take responsibility for the need to fund their performance development, which was identified in its own review.\n \"With UK Sport's investment budget approaching \u00a3350m, it borders on intransigence to pass responsibility to government and other funding bodies who are not set up to fund the development of high-performance sport.\"\n UK Sport says investment in the five Olympic sports and two Paralympic sports added for Tokyo 2020 is yet to be confirmed.\n Baseball/softball will return to the programme, with karate, skateboard, sports climbing and surfing also added, while Para-taekwondo and Para-badminton join the Paralympic programme.\n UK Sport says funding will be determined \"following further exploration of medal potential\", with \u00a39m of the \u00a3345m total still to be allocated.\n Liam Carroll, head coach of the GB baseball team, said: \"The key to unlocking our potential is investment and I'm pleased that UK Sport has left the door open.\n \"We look forward to the opportunity to impress upon them that getting behind Great Britain Baseball can extend their tremendous track record of investing in Olympic medal contenders.\"",
    "choice": "Badminton is one of five sports to lose all UK Sport funding for the 2020 Olympics in Tokyo - after Britain claimed a bronze in the sport in Rio."
}

BASIC_PROMPT_TEMPLATE = "[input]"
PROMPT_TEMPLATE_WITH_TXT_ON_BOTH_SIDES = "The summary of \"[input]\" is "
PROMPT_TEMPLATE_WITH_TXT_ON_LEFT_SIDE = "The summary of \"[input]"
PROMPT_TEMPLATE_WITH_TXT_ON_RIGHT_SIDE = "[input]\" is "


def test_tokenize_input(tokenizer):
    add_bosToken, add_eosToken = does_tokenizer_addBosEosTokens(tokenizer)


    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   SHORT_DATAPOINT,
                                                                                   BASIC_PROMPT_TEMPLATE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_equality(reconstructed_inputTxt, SHORT_DATAPOINT["input"])

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   SHORT_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_BOTH_SIDES,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_equality(reconstructed_inputTxt, f"The summary of \"{SHORT_DATAPOINT['input']}\" is ")

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   SHORT_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_LEFT_SIDE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_equality(reconstructed_inputTxt, f"The summary of \"{SHORT_DATAPOINT['input']}")

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   SHORT_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_RIGHT_SIDE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_equality(reconstructed_inputTxt, f"{SHORT_DATAPOINT['input']}\" is ")


    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   LONG_DATAPOINT,
                                                                                   BASIC_PROMPT_TEMPLATE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_subset_of_another(reconstructed_inputTxt, LONG_DATAPOINT["input"])

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   LONG_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_BOTH_SIDES,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_subset_of_another(reconstructed_inputTxt\
                                   .replace("The summary of \"", "")\
                                   .replace("\" is ", ""),
                                   LONG_DATAPOINT["input"])
    check_string_starts_with_another(reconstructed_inputTxt, "The summary of \"")
    check_string_ends_with_another(reconstructed_inputTxt, "\" is ")

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   LONG_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_LEFT_SIDE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_subset_of_another(reconstructed_inputTxt\
                                   .replace("The summary of \"", ""),
                                   LONG_DATAPOINT["input"])
    check_string_starts_with_another(reconstructed_inputTxt, "The summary of \"")

    input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(tokenizer,
                                                                                   LONG_DATAPOINT,
                                                                                   PROMPT_TEMPLATE_WITH_TXT_ON_RIGHT_SIDE,
                                                                                   add_bosToken,
                                                                                   add_eosToken)
    print("Length of Input Ids: ", len(input_ids))
    if add_bosToken:
        assert input_ids[0] == tokenizer.bos_token_id
    if add_eosToken:
        assert input_ids[0] == tokenizer.eos_token_id
    reconstructed_inputTxt = tokenizer.decode(input_ids, skip_special_tokens=True)
    check_string_equality(reconstructed_inputTxt, input_txt)
    check_string_subset_of_another(reconstructed_inputTxt\
                                   .replace("\" is ", ""),
                                   LONG_DATAPOINT["input"])
    check_string_ends_with_another(reconstructed_inputTxt, "\" is ")

if __name__ == "__main__":

    for tokenizer_name in ["bigscience/bloom-560m",
                           "facebook/opt-125m",
                           "gpt2-xl"]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.max_seq_len = 512
        test_tokenize_input(tokenizer)