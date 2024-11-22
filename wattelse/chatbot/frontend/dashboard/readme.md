


## structure of the raw_data csv 

dataframe, with columns : Index(['username', 'group_id', 'conversation_id', 'message', 'response',
       'answer_timestamp', 'answer_delay', 'short_feedback', 'long_feedback',
       'relevant_extracts']


within the raw_data csv, structure of the dict fo dict of dict relevant_extracts : 
{num_chunks :
	content
	{metadata :
		author
		creationDate
		creator
		file_name
		file_path
		format
		keywords
		modDate
		page
		producer
		source
		subject
		title
		total_pages
		trapped
    }
}


