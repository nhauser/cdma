#ifndef __SELECTION_HPP__
#define __SELECTION_HPP__

class Selection
{
    private:
        std::vector<size_t> _offset; //!< offset of the selection
        std::vector<size_t> _stride; //!< steps between elements
        std::vector<size_t> _shape;  //!< number of elements along each dimension
    public:
        //===================constructors and destructor=======================
        //! default constructor
        Selection():_offset(0),_stride(0),_shape(0) {}

        //----------------------------------------------------------------------
        //! copy constructor
        Selection(const Selection &s):
            _offset(s._offset),
            _stride(s._stride),
            _shape(s._shape)
        {}

        //----------------------------------------------------------------------
        //! move constructor
        Selection(Selection &&s):
            _offset(std::move(s._offset)),
            _stride(std::move(s._stride)),
            _shape(std::move(s._shape))
        {}

        //----------------------------------------------------------------------
        //! standard constructor
        Selection(const std::vector<size_t> &o,
                  const std::vector<size_t> &st,std::vector<size_t> sh):
            _offset(o),
            _stride(str),
            _shape(sh)
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~Selection(){}

        //===================assignment operators==============================
        //! copy assignment operator
        Selection &operator=(const Selection &s)
        {
            if(this == &s) return *this;
            _offset = s._offset;
            _stride = s._stride;
            _shape  = s._shape;
            return *this;
        }

        //---------------------------------------------------------------------
        //! move assignemnt operator
        Selection &operator=(Selection &&s)
        {
            if(this == &s) return *this;

            _offset = std::move(s._offset);
            _stride = std::move(s._stride);
            _shape  = std::move(s._shape);
            return *this;
        }

    
        std::vector<size_t> offset() const { return _offset; }
        std::vector<size_t> stride() const { return _stride; }
        std::vector<size_t> shape() const  { return _shape;  }
};

#endif
