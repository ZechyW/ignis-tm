test = """
<table style="width:100%">
    <tr>
        <th>Firstname</th>
        <th>Lastname</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>Find Documents that have Topic `topic_id` within its `n` most probable topics. Because topics are distributions over terms and documents are distributions over topics, documents don't belong to individual topics per se -- As such, we can consider a document to "belong" to a topic if it that topic is one of the `n` most probable topics for the document. This is especially significant if a term weighting scheme is used, because the most frequent words (i.e., what we usually consider stopwords) tend to be gathered into one or two topics, and we don't want to rule out a document if its single most probable topic is the stopword topic.
        </td>
        <td>Smith</td>
        <td>50</td>
    </tr>
    <tr>
        <td>Eve</td>
        <td>Jackson</td>
        <td>94</td>
    </tr>
</table>
"""

import dashtable

print(dashtable.html2rst(test))
