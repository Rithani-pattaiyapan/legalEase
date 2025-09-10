import mongoose,{Schema} from "mongoose";

const caseSchema = new Schema({
    title: {
        type: String,
        required: true,
        trim : true,
    },
    description: {
        type: String,
        required: true,
        trim : true,
    },
    legal_category: {   
        type: String,
        enum: ['criminal', 'civil', 'family', 'corporate', 'property', 'labor', 'tax', 'intellectual_property', 'environmental', 'human_rights', 'others'],
        required: true,
    },

    // Relations

    client_id: {
        type: Schema.Types.ObjectId,
        ref: 'CommonUser',
        required: true,
    },
    lawyer_id: {
        type: Schema.Types.ObjectId,
        ref: 'LawyerProfile',
        required: true,
    },
    status: {
        type: String,
        enum: ['ongoing', 'completed', 'closed'],
        default: 'ongoing',
    },

    process_updates: [
        {
            update_text: {
                type: String,
                required: true,
            },
            updated_by: {
                type: Schema.Types.ObjectId,
                ref: 'LawyerProfile',
            },
            timestamp: {
                type: Date,
                default: Date.now,
            },
        }
    ],

    created_at: {
        type: Date,
        default: Date.now,
    },
    updated_at: {
        type: Date,
        default: Date.now,
    },
});

caseSchema.pre('save', function (next) {
    this.updated_at = Date.now();
    next();
});

const Case = mongoose.model('Case', caseSchema);

export default Case; 