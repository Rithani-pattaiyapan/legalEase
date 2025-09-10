import mongoose,{Schema} from 'mongoose';

const userSchema = new Schema({
    name: {
        type: String,
        required: true,
        trim : true,
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim : true,    
        lowercase: true,
    },
    password_hash: {
        type: String,
        required: true,
    },
    role: {
        type: String,
        enum: ['common', 'lawyer','student', 'admin'],
        default: 'common',
    },
    phone : {       
        type: String,
        required: false,
    },
    profile_picture: {
        type: String,
        required: false,
    },
    created_at: {
        type: Date,
        default: Date.now,
    },
    updated_at: {
        type: Date,
        default: Date.now,
    },
});

userSchema.pre('save', function (next) {
    this.updated_at = Date.now();
    next();
});

const User = mongoose.model('User', userSchema);

export default User;