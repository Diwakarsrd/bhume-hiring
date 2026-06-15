# AI Transcripts

## Web chat transcripts

This solution was developed with Claude.ai assistance. The conversation covering:
- Problem analysis and method design
- IDW interpolation approach selection
- Confidence calibration strategy
- Code review and debugging

**Share link:** *(add your Claude.ai conversation share link here after submitting)*

To get the link: open the conversation in Claude.ai → Share → Copy public link.

## What AI was used for

1. **Problem framing**: Discussed the georeferencing error model (paper → scan → georeference)
   and why global shift is a weak baseline.

2. **Method selection**: Evaluated IDW vs kriging vs affine transform. IDW chosen because:
   - Only 6 truths in Vadnerbhairav — too few to fit kriging variogram reliably
   - IDW degrades gracefully (falls back toward global average far from truths)
   - Interpretable and debuggable

3. **Confidence design**: Reasoned through what signals genuinely predict accuracy
   (distance to nearest truth, local shift variance) vs. signals that look good but
   don't generalise (overfitting to the 6 known plots).

4. **Code**: predict.py written with AI assistance, then verified by running on real data.
