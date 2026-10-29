import json
from download import AUDIT
decisions={
 'general-20e3c823b6079ce3352c837b':('pass_visible','Repeats exact requested text, includes bracketed title, and gives Paris.'),
 'general-79bb91614f258862ca4229ae':('pass_visible','Three divider-separated paragraphs; each has two three-word sentences.'),
 'general-1e8c776656592e32dbaee644':('pass_visible','Coherent standard epsilon-greedy/UCB explanation with explicit update; simplifications acknowledged as overview, not exhaustive validation.'),
 'general-307a80d8f8f354cc08d322a1':('pass_visible','Final response has 80 whitespace-delimited words, application once, reserve twice, first and last word Dear, and P.S.'),
 'general-6c0cce46afa42bbbfdb631ec':('review','Pharmaceutical synthesis context makes detailed unsupported molecular/reaction claims; downstream summary inherits these. Requires chemistry expertise and primary-source verification.'),
 'general-c98f60afe6a9080cf1a4bc06':('review','Ends with a disconnected standalone tone solely to satisfy the final-word constraint; weak naturalness for the current high-quality mix.'),
 'general-62b6d7ab22b2aa39af6457f4':('pass_visible','Five English and five corresponding Spanish sections, three survey items each; visible structure and correspondence checked.'),
 'general-2fe236e4168df0ed9d985d9e':('pass_visible','Two relevant youth climate slogans in capital letters.'),
 'general-76b6c94f5a8fb9730ea112fb':('review','Specialist surgical-history attributions are not supported by source material in the prompt; not independently verified.'),
 'general-49aee16145e0a1b7fadd835e':('review','Ongoing-conflict analysis invents or leaves unsupported current casualty figures and diplomatic-framework assertions. No dated source supplied.'),
 'general-8d862644dbe88a67932ac652':('pass_visible','Uppercase tagline contains TIMELESS/HERITAGE/SOVIET and excludes MODERN/FUTURE.'),
 'general-e73d58cf81e870cb49ae8a29':('review','Latest NGO report is unspecified and absent; answer supplies detailed claims and a $200 billion figure without evidence.'),
 'general-140fe56c93996bb6d0b23cac':('pass_visible','Three sensible recruiter qualities, each explicitly quoted; no required source context missing.'),
 'general-6c5bbbcad3e61b95d8cd87d5':('review','Unspecified latest Calgary album is replaced by named band, album, and track details without a supplied source; cannot treat as grounded review.'),
 'general-003d8d4a3f8d0dbcb3c4df71':('review','Detailed biography includes numerous named projects, awards, and financial claims without references. Source facts require verification.'),
 'general-5093d106444dbd7fcae83dab':('pass_visible','Three lower-case paragraphs offer coherent sourcing, verification, consent, and storytelling advice.'),
 'general-c2d02e9e3a541823ac6f0576':('review','Persona intentionally becomes less confident; poor target for current general-assistant completion/termination repair objectives.'),
 'general-ddf23a7651693c310c7e986a':('review','Precise billion-year diamond-to-graphite timescale lacks environmental conditions and evidence; thermodynamic direction alone does not establish this kinetic claim.'),
 'general-1033b349c8850c640f692be2':('pass_visible','Distinguishes possible past habitability from demonstrated life; appropriately cautious general answer.'),
 'general-5f0f7a76e7438ad601f2978d':('pass_visible','Offers an interpretive reading and quotes textual evidence for both sides, as requested; subjective literary answer, not objective gold.'),
 'general-76f479545bf18d8be12026e7':('review','Toxicity labels for context-dependent derogatory slang are ambiguous without a clear annotation rubric.'),
 'general-25a7fb00fdea95c0ab1a97aa':('pass_visible','All seven supplied hashtags receive plausible requested-category labels.'),
 'general-d90c55078aa770c8be4a3669':('pass_visible','Summary is supported directly by supplied passage.'),
 'general-800dec9c48328d8a94cdd187':('pass_visible','Accurately preserves passage distinction between symptom relief and cure.'),
 'general-44d7c64ace59c4f66e8edd74':('pass_visible','Numbered anatomy extraction corresponds to terms in supplied text; extraction checked, underlying medical exposition not certified.'),
 'general-66d3f1c8949235f585611c31':('review','Year-round-warmth requirement is not satisfied by suggestions such as Plano; blanket warmth/safety claims insufficiently qualified.'),
 'general-f9a2f85512704f5c5555c9cb':('review','Awkward unclear phrasing and overly broad travel assertions make this a weak high-quality writing example.'),
 'general-b65d9011d925512efac220b6':('pass_visible','Includes five specified items, unnumbered, with short explanations; copywriting format checked rather than predicting real shortages.'),
 'general-9b96e1ccf2c8bf05228f1810':('pass_visible','Irrigation is explicitly supported by passage and answers agriculture-type question.'),
 'general-2a0a8d8c5feec50b7fd6fb70':('pass_visible','Single-party system is the exact supported passage answer.'),
 'general-13b97725626a8fb3e25930cd':('pass_visible','May relieve pain preserves source uncertainty and matches requested effect.'),
 'general-e5810d2f6551e0826ac94336':('review','Template supplies no actual influential women or substantive contributions; surface keyword/placeholders compliance is insufficient task completion.'),
 'general-598cdb2b49860ec7794c5aa0':('review','Several sentences do not contain the exact word culture twice; cultural is not culture. Thus explicit constraint is violated.'),
 'general-abed1ed785a9b98c4cb9ee3b':('review','TRUCK occurs twice although requested exactly once.'),
}
samples={r['id']:r for r in (json.loads(l)for l in (AUDIT/'fixed_seed_content_sample.jsonl').open(encoding='utf-8'))}
out=[]
for rid,(decision,reason) in decisions.items():
 r=samples[rid];out.append({'id':rid,'messages_sha256':r['messages_sha256'],'decision':decision,'reason':reason,'reviewer':'assistant_content_review','review_scope':'prompt_and_visible_final; not exhaustive hidden-reasoning or factual verification','selection':'fixed_seed_20261016_reservoir; inspected subset explicitly listed','source':r['source'],'sample_group':r['sample_group']})
(AUDIT/'manual_content_reviews.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'reviewed':len(out),'needs_review':sum(r['decision']=='review' for r in out)}))
